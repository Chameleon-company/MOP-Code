"""
Geometry utilities for pathway analysis.
Operates on GeoJSON LineString geometries.

All distance calculations use the haversine formula for geographic coordinates.
"""

import math
from typing import Optional


EARTH_RADIUS_M = 6_371_000


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def measure_polyline_length(coords: list[tuple]) -> float:
    """
    Sum haversine distances between consecutive coordinate pairs.

    Args:
        coords: List of (lat, lon) tuples.

    Returns:
        Total length in metres.
    """
    total = 0.0
    for i in range(len(coords) - 1):
        total += _haversine(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return total


def _point_near(p1: tuple, p2: tuple, threshold_m: float) -> bool:
    """Return True if two (lat, lon) points are within threshold_m of each other."""
    return _haversine(p1[0], p1[1], p2[0], p2[1]) < threshold_m


def find_intersections(pathways: list[dict], threshold_m: float = 5.0) -> list[dict]:
    """
    Find points where pathway polylines cross or share nodes.

    Args:
        pathways: List of dicts with "geometry" containing GeoJSON LineString
                  (coordinates as [[lon, lat], ...]).
        threshold_m: Distance threshold to consider two nodes as the same point.

    Returns:
        List of intersection dicts: {"lat": float, "lng": float, "paths_meeting": int}
    """
    # Collect all nodes from all pathways, keyed by (lat, lon)
    node_counts: dict[tuple, int] = {}

    for pw in pathways:
        geom = pw.get("geometry", {})
        coords = geom.get("coordinates", [])
        seen_in_this_path = set()
        for coord in coords:
            lon, lat = coord[0], coord[1]
            point = (round(lat, 7), round(lon, 7))
            # Check if this point is near an existing node
            matched = False
            for existing in list(node_counts.keys()):
                if _point_near(point, existing, threshold_m) and existing not in seen_in_this_path:
                    node_counts[existing] += 1
                    seen_in_this_path.add(existing)
                    matched = True
                    break
            if not matched and point not in seen_in_this_path:
                node_counts[point] = node_counts.get(point, 0) + 1
                seen_in_this_path.add(point)

    # Intersections are nodes shared by 2+ pathways
    intersections = []
    for (lat, lon), count in node_counts.items():
        if count >= 2:
            intersections.append({"lat": lat, "lng": lon, "paths_meeting": count})

    return intersections


def find_entry_points(
    pathway: dict, park_boundary: dict, threshold_m: float = 20.0
) -> list[dict]:
    """
    Find where pathway endpoints are near (< threshold_m) the park boundary.

    Args:
        pathway: Dict with "geometry" (GeoJSON LineString).
        park_boundary: GeoJSON Polygon dict with "coordinates" as [[[lon, lat], ...]].
        threshold_m: Max distance to consider an entry point.

    Returns:
        List of entry point dicts: {"lat": float, "lng": float, "adjacent_to": str}
    """
    geom = pathway.get("geometry", {})
    path_coords = geom.get("coordinates", [])
    if not path_coords:
        return []

    # Get boundary ring coordinates
    boundary_coords = []
    raw_boundary = park_boundary.get("coordinates", [])
    if raw_boundary and isinstance(raw_boundary[0], list):
        # Polygon: first element is the outer ring
        ring = raw_boundary[0] if isinstance(raw_boundary[0][0], list) else raw_boundary
        for coord in ring:
            boundary_coords.append((coord[1], coord[0]))  # lat, lon

    if not boundary_coords:
        return []

    entry_points = []
    # Check start and end of pathway
    endpoints = [path_coords[0], path_coords[-1]]
    for coord in endpoints:
        lon, lat = coord[0], coord[1]
        for blat, blon in boundary_coords:
            if _haversine(lat, lon, blat, blon) < threshold_m:
                entry_points.append({
                    "lat": lat,
                    "lng": lon,
                    "adjacent_to": "park_boundary",
                })
                break

    return entry_points


def _interpolate_point(p1: tuple, p2: tuple, fraction: float) -> tuple:
    """Linearly interpolate between two (lat, lon) points."""
    return (
        p1[0] + (p2[0] - p1[0]) * fraction,
        p1[1] + (p2[1] - p1[1]) * fraction,
    )


def place_lights_on_polyline(
    coords: list[tuple],
    spacing_m: float,
    intersections: Optional[list[dict]] = None,
    entry_points: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Place lights along a pathway polyline at regular intervals.

    Mandatory placements: every intersection, every entry point.
    Fill remaining segments at the calculated spacing.

    Args:
        coords: List of (lat, lon) tuples defining the pathway.
        spacing_m: Target spacing between lights in metres.
        intersections: List of intersection dicts with "lat" and "lng".
        entry_points: List of entry point dicts with "lat" and "lng".

    Returns:
        List of light placement dicts:
        {"lat": float, "lng": float, "type": str, "chainage_m": float}
    """
    if not coords or spacing_m <= 0:
        return []

    intersections = intersections or []
    entry_points = entry_points or []

    # Build a chainage map: distance along the polyline for each vertex
    chainages = [0.0]
    for i in range(1, len(coords)):
        seg = _haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        chainages.append(chainages[-1] + seg)
    total_length = chainages[-1]

    if total_length == 0:
        return [{"lat": coords[0][0], "lng": coords[0][1], "type": "pathway", "chainage_m": 0.0}]

    # Collect mandatory placements with their chainage
    mandatory = []

    # Find chainage for intersection/entry points by projecting onto polyline
    for pt_list, pt_type in [(intersections, "intersection"), (entry_points, "entry")]:
        for pt in pt_list:
            plat, plon = pt["lat"], pt["lng"]
            # Project point onto nearest segment for accurate chainage
            best_chainage = 0.0
            best_dist = float("inf")
            for i in range(len(coords) - 1):
                # Check projection onto segment i -> i+1
                seg_len = chainages[i + 1] - chainages[i]
                if seg_len == 0:
                    continue
                d_start = _haversine(plat, plon, coords[i][0], coords[i][1])
                d_end = _haversine(plat, plon, coords[i + 1][0], coords[i + 1][1])
                # Approximate projection fraction along segment
                # Use cosine rule to find along-segment distance
                if d_start < best_dist:
                    best_dist = d_start
                    best_chainage = chainages[i]
                if d_end < best_dist:
                    best_dist = d_end
                    best_chainage = chainages[i + 1]
                # Check perpendicular projection
                frac = max(0.0, min(1.0, (d_start**2 + seg_len**2 - d_end**2) / (2 * seg_len**2) * seg_len))
                proj_lat = coords[i][0] + (coords[i + 1][0] - coords[i][0]) * (frac / seg_len if seg_len else 0)
                proj_lon = coords[i][1] + (coords[i + 1][1] - coords[i][1]) * (frac / seg_len if seg_len else 0)
                d_proj = _haversine(plat, plon, proj_lat, proj_lon)
                if d_proj < best_dist:
                    best_dist = d_proj
                    best_chainage = chainages[i] + frac
            # Also check last vertex
            d_last = _haversine(plat, plon, coords[-1][0], coords[-1][1])
            if d_last < best_dist:
                best_dist = d_last
                best_chainage = chainages[-1]
            # Include if within reasonable distance of the polyline
            if best_dist < total_length * 0.5:
                mandatory.append({"lat": plat, "lng": plon, "type": pt_type, "chainage_m": round(best_chainage, 1)})

    # Generate regular spacing placements
    regular = []
    chainage = 0.0
    while chainage <= total_length:
        # Find the position on the polyline at this chainage
        lat, lon = _point_at_chainage(coords, chainages, chainage)
        regular.append({"lat": lat, "lng": lon, "type": "pathway", "chainage_m": round(chainage, 1)})
        chainage += spacing_m

    # Ensure a light at the end
    if regular and abs(regular[-1]["chainage_m"] - total_length) > spacing_m * 0.3:
        lat, lon = coords[-1]
        regular.append({"lat": lat, "lng": lon, "type": "pathway", "chainage_m": round(total_length, 1)})

    # Merge: mandatory placements override nearby regular placements
    merged = list(mandatory)
    mandatory_chainages = {m["chainage_m"] for m in mandatory}
    for r in regular:
        # Skip if a mandatory placement is within half-spacing
        too_close = any(abs(r["chainage_m"] - mc) < spacing_m * 0.4 for mc in mandatory_chainages)
        if not too_close:
            merged.append(r)

    # Sort by chainage
    merged.sort(key=lambda x: x["chainage_m"])
    return merged


def _point_at_chainage(
    coords: list[tuple], chainages: list[float], target: float
) -> tuple:
    """Find the (lat, lon) point at a given chainage along the polyline."""
    if target <= 0:
        return coords[0]
    if target >= chainages[-1]:
        return coords[-1]

    for i in range(1, len(chainages)):
        if chainages[i] >= target:
            seg_length = chainages[i] - chainages[i - 1]
            if seg_length == 0:
                return coords[i]
            fraction = (target - chainages[i - 1]) / seg_length
            return _interpolate_point(coords[i - 1], coords[i], fraction)

    return coords[-1]
