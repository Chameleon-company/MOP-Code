"""
OpenStreetMap data loader for Melbourne parks and pathways.
Fetches park boundaries and pathway geometries via Overpass API,
caches results locally as GeoJSON.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

from data.geometry import measure_polyline_length, find_intersections, find_entry_points

CACHE_DIR = Path(__file__).parent / "cache" / "osm"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CACHE_MAX_AGE_DAYS = 30


def _slugify(name: str) -> str:
    """Convert a park name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _cache_path(park_name: str, suffix: str = "") -> Path:
    slug = _slugify(park_name)
    return CACHE_DIR / f"{slug}{suffix}.json"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_days = (time.time() - path.stat().st_mtime) / 86400
    return age_days < CACHE_MAX_AGE_DAYS


def _save_cache(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _load_cache(path: Path) -> dict:
    return json.loads(path.read_text())


def _query_overpass(query: str, timeout: int = 60) -> dict:
    """Execute an Overpass API query and return the JSON response."""
    resp = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=(15, timeout + 15),
    )
    resp.raise_for_status()
    return resp.json()


def geocode_nominatim(place_name: str, city: str = "Melbourne") -> Optional[dict]:
    """
    Lightweight geocoding fallback via Nominatim.
    Returns a park_boundary-compatible dict derived from the bounding box,
    or None if the place cannot be found.

    Used when Overpass is unavailable — gives us bounds for spatial context
    but no pathway geometry.
    """
    cache = _cache_path(place_name, "_nominatim")
    if _is_cache_fresh(cache):
        return _load_cache(cache)

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": f"{place_name}, {city}, Australia",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": "SmartStreetLighting/1.0 (university-capstone)"},
            timeout=(10, 15),
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        print(f"Nominatim geocoding failed for '{place_name}': {e}")
        return None

    if not results:
        print(f"Nominatim found no results for '{place_name}' in {city}.")
        return None

    hit = results[0]
    bbox = hit.get("boundingbox")  # [south_lat, north_lat, west_lon, east_lon] as strings
    if not bbox or len(bbox) < 4:
        return None

    south, north = float(bbox[0]), float(bbox[1])
    west, east = float(bbox[2]), float(bbox[3])

    # Build a rectangular GeoJSON polygon from the bounding box
    boundary = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }

    result = {
        "boundary": boundary,
        "name": hit.get("display_name", place_name),
        "osm_id": hit.get("osm_id"),
        "source": "nominatim",
    }

    _save_cache(cache, result)
    return result


def _nodes_to_coords(elements: list) -> dict:
    """Build a node_id -> (lat, lon) lookup from Overpass elements."""
    return {
        e["id"]: (e["lat"], e["lon"])
        for e in elements
        if e["type"] == "node" and "lat" in e and "lon" in e
    }


def _way_to_polygon(way: dict, node_lookup: dict) -> Optional[list]:
    """Convert a way's node refs to a coordinate ring [[lon, lat], ...]."""
    coords = []
    for nid in way.get("nodes", []):
        if nid in node_lookup:
            lat, lon = node_lookup[nid]
            coords.append([lon, lat])
    return coords if len(coords) >= 3 else None


def fetch_park_data(park_name: str, city: str = "Melbourne") -> Optional[dict]:
    """
    Fetch park boundary from Overpass API.

    Args:
        park_name: Name of the park (e.g., "Fitzroy Gardens").
        city: City to search within.

    Returns:
        Dict with "boundary" (GeoJSON Polygon), "name", "osm_id", or None if not found.
    """
    cache = _cache_path(park_name, "_park")
    if _is_cache_fresh(cache):
        return _load_cache(cache)

    query = f"""
    [out:json][timeout:60];
    area["name"="{city}"]->.city;
    (
      way(area.city)["leisure"="park"]["name"~"{park_name}",i];
      relation(area.city)["leisure"="park"]["name"~"{park_name}",i];
    );
    out body; >; out skel qt;
    """

    try:
        data = _query_overpass(query)
    except Exception as e:
        print(f"Overpass query failed for park '{park_name}': {e}")
        return None

    elements = data.get("elements", [])
    if not elements:
        print(f"No OSM park found for '{park_name}' in {city}.")
        return None

    node_lookup = _nodes_to_coords(elements)

    # Find the first way or relation that represents the park
    boundary_coords = None
    osm_id = None
    osm_name = park_name

    for el in elements:
        if el["type"] == "way" and "nodes" in el:
            tags = el.get("tags", {})
            if tags.get("leisure") == "park":
                coords = _way_to_polygon(el, node_lookup)
                if coords:
                    boundary_coords = coords
                    osm_id = el["id"]
                    osm_name = tags.get("name", park_name)
                    break

        elif el["type"] == "relation":
            tags = el.get("tags", {})
            if tags.get("leisure") == "park":
                # Relations: collect outer way members
                osm_id = el["id"]
                osm_name = tags.get("name", park_name)
                for member in el.get("members", []):
                    if member.get("role") == "outer" and member["type"] == "way":
                        # Find this way in elements
                        for w in elements:
                            if w["type"] == "way" and w["id"] == member["ref"]:
                                coords = _way_to_polygon(w, node_lookup)
                                if coords:
                                    boundary_coords = coords
                                    break
                    if boundary_coords:
                        break

    if not boundary_coords:
        print(f"Could not extract boundary for '{park_name}'.")
        return None

    result = {
        "boundary": {
            "type": "Polygon",
            "coordinates": [boundary_coords],
        },
        "name": osm_name,
        "osm_id": osm_id,
    }

    _save_cache(cache, result)
    return result


def fetch_pathways(park_boundary: dict) -> list[dict]:
    """
    Fetch pathways within a park boundary from Overpass API.

    Args:
        park_boundary: GeoJSON Polygon dict with "coordinates".

    Returns:
        List of pathway dicts with geometry, length, type, name, surface.
    """
    # Build poly string from boundary coords: "lat1 lon1 lat2 lon2 ..."
    ring = park_boundary.get("coordinates", [[]])[0]
    if not ring:
        return []

    poly_parts = []
    for coord in ring:
        lon, lat = coord[0], coord[1]
        poly_parts.append(f"{lat} {lon}")
    poly_str = " ".join(poly_parts)

    query = f"""
    [out:json][timeout:60];
    way(poly:"{poly_str}")["highway"~"footway|path|cycleway|pedestrian"];
    out geom;
    """

    try:
        data = _query_overpass(query)
    except Exception as e:
        print(f"Overpass pathway query failed: {e}")
        return []

    pathways = []
    for el in data.get("elements", []):
        if el["type"] != "way" or "geometry" not in el:
            continue

        coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
        lat_lon_coords = [(pt["lat"], pt["lon"]) for pt in el["geometry"]]
        length = measure_polyline_length(lat_lon_coords)

        tags = el.get("tags", {})
        pathways.append({
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "length_m": round(length, 1),
            "highway_type": tags.get("highway", "path"),
            "name": tags.get("name"),
            "surface": tags.get("surface"),
        })

    return pathways


def resolve_pathway(
    park_name: str, pathway_hint: Optional[str] = None
) -> Optional[dict]:
    """
    High-level function: resolve park boundary and pathways from OSM.

    Args:
        park_name: Name of the park.
        pathway_hint: Optional hint like "main pathway" to select a primary pathway.

    Returns:
        Dict with park_name, park_boundary, pathways, selected_pathway index.
        None if park not found.
    """
    full_cache = _cache_path(park_name, "_resolved")
    if _is_cache_fresh(full_cache):
        return _load_cache(full_cache)

    park = fetch_park_data(park_name)
    if not park:
        # Overpass failed — try Nominatim for boundary-only fallback
        print(f"Overpass unavailable for '{park_name}', trying Nominatim geocoding...")
        nominatim = geocode_nominatim(park_name)
        if nominatim:
            print(f"Nominatim resolved '{park_name}' (boundary only, no pathways).")
            result = {
                "park_name": nominatim["name"],
                "park_boundary": nominatim["boundary"],
                "pathways": [],
                "selected_pathway": None,
                "data_source": "nominatim",
            }
            _save_cache(full_cache, result)
            return result
        return None

    pathways_raw = fetch_pathways(park["boundary"])
    if not pathways_raw:
        print(f"No pathways found in '{park_name}'.")
        # Still return the boundary
        result = {
            "park_name": park["name"],
            "park_boundary": park["boundary"],
            "pathways": [],
            "selected_pathway": None,
        }
        _save_cache(full_cache, result)
        return result

    # Enrich each pathway with intersections and entry points
    intersections = find_intersections(pathways_raw)

    enriched_pathways = []
    longest_idx = 0
    longest_length = 0

    for i, pw in enumerate(pathways_raw):
        geom = pw["geometry"]
        lat_lon = [(c[1], c[0]) for c in geom["coordinates"]]
        true_length = measure_polyline_length(lat_lon)

        # Find entry points for this pathway
        entries = find_entry_points(pw, park["boundary"])

        # Find intersections that belong to this pathway
        pw_intersections = []
        for ix in intersections:
            for coord in geom["coordinates"]:
                if abs(coord[1] - ix["lat"]) < 0.0001 and abs(coord[0] - ix["lng"]) < 0.0001:
                    pw_intersections.append(ix)
                    break

        enriched_pathways.append({
            "geometry": geom,
            "true_length_m": round(true_length, 1),
            "intersections": pw_intersections,
            "entry_points": entries,
            "is_primary": False,
            "highway_type": pw["highway_type"],
            "name": pw.get("name"),
            "surface": pw.get("surface"),
        })

        if true_length > longest_length:
            longest_length = true_length
            longest_idx = i

    # Mark primary pathway
    if enriched_pathways:
        enriched_pathways[longest_idx]["is_primary"] = True

    # Select pathway based on hint
    selected = None
    if pathway_hint and enriched_pathways:
        hint = pathway_hint.lower()
        if "main" in hint or "primary" in hint or "longest" in hint:
            selected = longest_idx
        else:
            # Try name matching
            for i, pw in enumerate(enriched_pathways):
                if pw.get("name") and hint in pw["name"].lower():
                    selected = i
                    break
            if selected is None:
                selected = longest_idx
    elif enriched_pathways:
        selected = longest_idx

    result = {
        "park_name": park["name"],
        "park_boundary": park["boundary"],
        "pathways": enriched_pathways,
        "selected_pathway": selected,
    }

    _save_cache(full_cache, result)
    return result
