"""
Tests for OSM pathway geometry integration.
Uses cached fixture data — no network calls required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.geometry import (
    measure_polyline_length,
    find_intersections,
    find_entry_points,
    place_lights_on_polyline,
    _haversine,
)
from data.osm_loader import resolve_pathway, geocode_nominatim, _slugify
from core.pipeline import bounds_from_osm_boundary

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# Geometry: measure_polyline_length
# ============================================================

class TestMeasurePolylineLength:
    def test_simple_3_point_line(self):
        """Three points roughly 111m apart (0.001 degrees latitude at Melbourne)."""
        coords = [
            (-37.8100, 144.9800),
            (-37.8110, 144.9800),
            (-37.8120, 144.9800),
        ]
        length = measure_polyline_length(coords)
        # 0.002 degrees lat ~ 222m at Melbourne latitude
        assert 200 < length < 250, f"Expected ~222m, got {length}"

    def test_single_point(self):
        length = measure_polyline_length([(-37.81, 144.98)])
        assert length == 0.0

    def test_empty(self):
        assert measure_polyline_length([]) == 0.0

    def test_known_distance(self):
        """Flinders St Station to Federation Square is ~200m."""
        flinders = (-37.8183, 144.9671)
        fed_sq = (-37.8180, 144.9695)
        dist = _haversine(*flinders, *fed_sq)
        assert 180 < dist < 250


# ============================================================
# Geometry: place_lights_on_polyline
# ============================================================

class TestPlaceLights:
    def test_straight_line_spacing(self):
        """100m straight line with 20m spacing should give 6 lights (0, 20, 40, 60, 80, 100)."""
        # ~100m north-south line
        coords = [(-37.8100, 144.9800), (-37.8109, 144.9800)]
        length = measure_polyline_length(coords)
        assert 95 < length < 105

        lights = place_lights_on_polyline(coords, spacing_m=20.0)
        assert len(lights) == 6
        assert lights[0]["chainage_m"] == 0.0
        assert all(l["type"] == "pathway" for l in lights)

    def test_mandatory_intersection_placement(self):
        """Intersection point should appear in output even if not on regular spacing."""
        coords = [(-37.8100, 144.9800), (-37.8109, 144.9800)]
        intersections = [{"lat": -37.8105, "lng": 144.9800, "paths_meeting": 2}]

        lights = place_lights_on_polyline(coords, spacing_m=20.0, intersections=intersections)
        types = [l["type"] for l in lights]
        assert "intersection" in types

    def test_mandatory_entry_placement(self):
        """Entry points should appear in output."""
        coords = [(-37.8100, 144.9800), (-37.8109, 144.9800)]
        entry = [{"lat": -37.8100, "lng": 144.9800, "adjacent_to": "boundary"}]

        lights = place_lights_on_polyline(coords, spacing_m=20.0, entry_points=entry)
        types = [l["type"] for l in lights]
        assert "entry" in types

    def test_chainage_monotonically_increasing(self):
        coords = [
            (-37.8100, 144.9800),
            (-37.8105, 144.9803),
            (-37.8110, 144.9800),
        ]
        lights = place_lights_on_polyline(coords, spacing_m=15.0)
        chainages = [l["chainage_m"] for l in lights]
        assert chainages == sorted(chainages)

    def test_minimum_two_lights(self):
        """Even a very short path should get at least a start and end light."""
        coords = [(-37.8100, 144.9800), (-37.81001, 144.9800)]
        lights = place_lights_on_polyline(coords, spacing_m=20.0)
        assert len(lights) >= 1


# ============================================================
# Geometry: find_intersections
# ============================================================

class TestFindIntersections:
    def test_crossing_paths(self):
        """Two paths sharing a node should produce an intersection."""
        pathways = [
            {"geometry": {"type": "LineString", "coordinates": [
                [144.9800, -37.8100], [144.9805, -37.8105], [144.9810, -37.8110],
            ]}},
            {"geometry": {"type": "LineString", "coordinates": [
                [144.9810, -37.8100], [144.9805, -37.8105], [144.9800, -37.8110],
            ]}},
        ]
        intersections = find_intersections(pathways)
        assert len(intersections) >= 1
        assert intersections[0]["paths_meeting"] >= 2

    def test_no_intersection(self):
        """Parallel paths with no shared nodes."""
        pathways = [
            {"geometry": {"type": "LineString", "coordinates": [
                [144.9800, -37.8100], [144.9810, -37.8100],
            ]}},
            {"geometry": {"type": "LineString", "coordinates": [
                [144.9800, -37.8200], [144.9810, -37.8200],
            ]}},
        ]
        intersections = find_intersections(pathways)
        assert len(intersections) == 0


# ============================================================
# OSM Loader: resolve_pathway with fixture
# ============================================================

class TestResolvePathway:
    def test_resolve_from_fixture(self):
        """Test resolve_pathway returns correct structure from cached fixture."""
        fixture = json.loads((FIXTURES_DIR / "fitzroy_gardens_resolved.json").read_text())

        with patch("data.osm_loader._is_cache_fresh", return_value=True), \
             patch("data.osm_loader._load_cache", return_value=fixture):
            result = resolve_pathway("Fitzroy Gardens")

        assert result is not None
        assert result["park_name"] == "Fitzroy Gardens"
        assert result["park_boundary"]["type"] == "Polygon"
        assert len(result["pathways"]) == 2
        assert result["pathways"][0]["is_primary"] is True
        assert result["selected_pathway"] == 0

    def test_resolve_returns_none_when_both_services_fail(self):
        """If both Overpass and Nominatim fail, should return None."""
        with patch("data.osm_loader._is_cache_fresh", return_value=False), \
             patch("data.osm_loader._query_overpass", return_value={"elements": []}), \
             patch("data.osm_loader.geocode_nominatim", return_value=None):
            result = resolve_pathway("Nonexistent Park XYZ")

        assert result is None

    def test_resolve_falls_back_to_nominatim(self):
        """If Overpass fails, Nominatim should provide boundary-only result."""
        nominatim_result = {
            "boundary": {
                "type": "Polygon",
                "coordinates": [[
                    [144.979, -37.816],
                    [144.983, -37.816],
                    [144.983, -37.812],
                    [144.979, -37.812],
                    [144.979, -37.816],
                ]],
            },
            "name": "Fitzroy Gardens, Melbourne",
            "osm_id": 12345,
            "source": "nominatim",
        }

        with patch("data.osm_loader._is_cache_fresh", return_value=False), \
             patch("data.osm_loader._query_overpass", return_value={"elements": []}), \
             patch("data.osm_loader.geocode_nominatim", return_value=nominatim_result):
            result = resolve_pathway("Fitzroy Gardens")

        assert result is not None
        assert result["park_boundary"]["type"] == "Polygon"
        assert result["pathways"] == []
        assert result["data_source"] == "nominatim"


# ============================================================
# Nominatim geocoding fallback
# ============================================================

class TestGeocodeNominatim:
    def _mock_nominatim_response(self, bbox, display_name="Fitzroy Gardens", osm_id=999):
        """Build a mock requests.get response for Nominatim."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{
            "boundingbox": bbox,
            "display_name": display_name,
            "osm_id": osm_id,
        }]
        return mock_resp

    def test_returns_boundary_from_bbox(self):
        bbox = ["-37.816", "-37.812", "144.979", "144.983"]
        with patch("data.osm_loader._is_cache_fresh", return_value=False), \
             patch("data.osm_loader.requests.get", return_value=self._mock_nominatim_response(bbox)):
            result = geocode_nominatim("Fitzroy Gardens")

        assert result is not None
        assert result["boundary"]["type"] == "Polygon"
        assert result["source"] == "nominatim"

        # Verify bounding box was converted to polygon correctly
        ring = result["boundary"]["coordinates"][0]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        assert min(lats) == pytest.approx(-37.816, abs=1e-3)
        assert max(lats) == pytest.approx(-37.812, abs=1e-3)
        assert min(lons) == pytest.approx(144.979, abs=1e-3)
        assert max(lons) == pytest.approx(144.983, abs=1e-3)

    def test_returns_none_when_no_results(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []
        with patch("data.osm_loader._is_cache_fresh", return_value=False), \
             patch("data.osm_loader.requests.get", return_value=mock_resp):
            result = geocode_nominatim("Nonexistent Place")

        assert result is None

    def test_returns_none_on_network_error(self):
        with patch("data.osm_loader._is_cache_fresh", return_value=False), \
             patch("data.osm_loader.requests.get", side_effect=ConnectionError("timeout")):
            result = geocode_nominatim("Fitzroy Gardens")

        assert result is None

    def test_uses_cache_when_fresh(self):
        cached = {"boundary": {"type": "Polygon", "coordinates": [[]]}, "name": "cached", "source": "nominatim"}
        with patch("data.osm_loader._is_cache_fresh", return_value=True), \
             patch("data.osm_loader._load_cache", return_value=cached):
            result = geocode_nominatim("Fitzroy Gardens")

        assert result["name"] == "cached"


# ============================================================
# bounds_from_osm_boundary
# ============================================================

class TestBoundsFromOsmBoundary:
    def test_extracts_correct_bounds_from_fixture(self):
        """Verify bounds match the min/max of the fixture polygon."""
        fixture = json.loads((FIXTURES_DIR / "fitzroy_gardens_resolved.json").read_text())
        bounds = bounds_from_osm_boundary(fixture["park_boundary"])

        assert bounds is not None
        lat_min, lat_max, lon_min, lon_max = bounds
        # Fixture ring: lon 144.9790–144.9820, lat -37.8155–-37.8128
        assert lat_min == pytest.approx(-37.8155, abs=1e-4)
        assert lat_max == pytest.approx(-37.8128, abs=1e-4)
        assert lon_min == pytest.approx(144.9790, abs=1e-4)
        assert lon_max == pytest.approx(144.9820, abs=1e-4)

    def test_returns_none_for_empty_boundary(self):
        assert bounds_from_osm_boundary({}) is None

    def test_returns_none_for_malformed_coordinates(self):
        assert bounds_from_osm_boundary({"coordinates": []}) is None

    def test_returns_none_for_none_input(self):
        assert bounds_from_osm_boundary(None) is None

    def test_irregular_polygon(self):
        """Non-rectangular polygon should still give correct min/max."""
        boundary = {
            "type": "Polygon",
            "coordinates": [[
                [144.95, -37.82],
                [144.98, -37.80],
                [144.96, -37.79],
                [144.94, -37.81],
                [144.95, -37.82],
            ]]
        }
        bounds = bounds_from_osm_boundary(boundary)
        lat_min, lat_max, lon_min, lon_max = bounds
        assert lat_min == pytest.approx(-37.82, abs=1e-4)
        assert lat_max == pytest.approx(-37.79, abs=1e-4)
        assert lon_min == pytest.approx(144.94, abs=1e-4)
        assert lon_max == pytest.approx(144.98, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
