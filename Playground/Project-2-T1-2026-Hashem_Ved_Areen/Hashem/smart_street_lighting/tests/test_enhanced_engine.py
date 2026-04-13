"""
Tests for enhanced calculation engine: geometry-aware placement + budget constraints.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from llm.calculation_engine import design_lighting


# ============================================================
# Geometry-aware placement
# ============================================================

class TestGeometryAwarePlacement:
    def test_with_geometry_produces_positions(self):
        """When pathway geometry is provided, light_positions should be populated."""
        geom = {
            "type": "LineString",
            "coordinates": [
                [144.9795, -37.8130],
                [144.9800, -37.8135],
                [144.9805, -37.8140],
                [144.9810, -37.8145],
                [144.9815, -37.8150],
            ],
        }
        d = design_lighting(
            "Test Park", 310, 3.0, "high", "park_path",
            pathway_geometry=geom,
        )
        assert len(d.light_positions) > 0
        assert all("lat" in p and "lng" in p for p in d.light_positions)
        assert all("chainage_m" in p for p in d.light_positions)

    def test_geometry_updates_light_count(self):
        """Geometry-placed light count should match the positions list."""
        geom = {
            "type": "LineString",
            "coordinates": [
                [144.9795, -37.8130],
                [144.9815, -37.8150],
            ],
        }
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path", pathway_geometry=geom)
        assert d.num_lights == len(d.light_positions)

    def test_without_geometry_no_positions(self):
        """Without geometry, light_positions should be empty."""
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        assert d.light_positions == []

    def test_intersection_lights_included(self):
        """Intersections should appear as placed lights."""
        geom = {
            "type": "LineString",
            "coordinates": [
                [144.9795, -37.8130],
                [144.9805, -37.8140],
                [144.9815, -37.8150],
            ],
        }
        intersections = [{"lat": -37.8140, "lng": 144.9805, "paths_meeting": 2}]
        d = design_lighting(
            "Test", 310, 3.0, "high", "park_path",
            pathway_geometry=geom, intersections=intersections,
        )
        types = [p["type"] for p in d.light_positions]
        assert "intersection" in types


# ============================================================
# Budget constraints
# ============================================================

class TestBudgetConstraints:
    def test_within_budget(self):
        """High budget should pass."""
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path", budget_cap=10000)
        assert d.budget_analysis["within_budget"] is True

    def test_over_budget_generates_alternative(self):
        """Very tight budget should produce an alternative."""
        d = design_lighting("Test", 200, 3.0, "high", "park_path", budget_cap=10)
        assert d.budget_analysis["within_budget"] is False
        assert d.budget_analysis["budget_alternative"] is not None

    def test_budget_alternative_has_wider_spacing(self):
        """Alternative should have wider spacing than primary design."""
        d = design_lighting("Test", 200, 3.0, "high", "park_path", budget_cap=10)
        alt = d.budget_analysis["budget_alternative"]
        assert alt["spacing_m"] > d.spacing_m

    def test_no_budget_no_analysis(self):
        """Without budget_cap, budget_analysis should be empty."""
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        assert d.budget_analysis == {}


# ============================================================
# Safety adjustment recorded
# ============================================================

class TestSafetyAdjustmentRecorded:
    def test_safety_adjustment_stored(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path", safety_adjustment=-1)
        assert d.safety_adjustment_applied == -1

    def test_summary_includes_safety(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path", safety_adjustment=-2)
        summary = d.summary_dict()
        assert summary["safety_adjustment_applied"] == -2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
