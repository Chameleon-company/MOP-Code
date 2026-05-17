"""
Shared pipeline logic for the Smart Street Lighting Design System.

Centralises data loading, query parsing, spatial context lookup,
and the design orchestration flow used by both the CLI and the API.
"""

import json
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_PARSE_MODEL,
    FAST_LLM_TIMEOUT,
    PARSE_MAX_TOKENS,
    FAST_LLM_TEMPERATURE,
)

from data.load_melbourne_data import (
    load_pedestrian_data,
    load_streetlight_data,
    get_sensor_summary,
)
from data.spatial_analysis import (
    match_sensors_to_lights,
    analyze_lighting_efficiency,
    get_area_lighting_context,
)
from data.temporal_analysis import (
    add_temporal_features,
    get_hourly_profile,
    suggest_dimming_schedule,
    estimate_dimming_savings,
    get_weekday_weekend_profile,
)
from llm.calculation_engine import design_lighting, format_design_report, P_CATEGORIES


# ============================================================
# Known Melbourne locations (for LLM parser hints only)
# ============================================================

KNOWN_LOCATIONS = [
    "fitzroy gardens",
    "royal park",
    "princes park",
    "carlton gardens",
    "flagstaff gardens",
    "birrarung marr",
    "edinburgh gardens",
    "treasury gardens",
    "yarra park",
    "fawkner park",
    "albert park",
]


def bounds_from_osm_boundary(
    boundary: dict,
) -> tuple[float, float, float, float] | None:
    """
    Extract a (lat_min, lat_max, lon_min, lon_max) bounding box from a
    GeoJSON Polygon returned by the OSM loader.

    Returns None if the boundary is missing or malformed.
    """
    try:
        ring = boundary["coordinates"][0]  # outer ring: [[lon, lat], ...]
        lats = [pt[1] for pt in ring]
        lons = [pt[0] for pt in ring]
        return (min(lats), max(lats), min(lons), max(lons))
    except (KeyError, IndexError, TypeError):
        return None


# ============================================================
# Data loading — call once at startup
# ============================================================


class AppData:
    """Holds all Melbourne data loaded at startup. Created once, shared across requests."""

    def __init__(self, ped_limit: int = 2000, light_limit: int = 5000):
        print("Loading Melbourne Open Data...")
        self.ped_data = load_pedestrian_data(limit=ped_limit)
        self.streetlight_data = load_streetlight_data(limit=light_limit)
        self.sensor_summary = get_sensor_summary(self.ped_data)

        print("Running spatial analysis...")
        self.ped_matched = match_sensors_to_lights(
            self.ped_data.copy(), self.streetlight_data
        )
        self.ped_matched = analyze_lighting_efficiency(self.ped_matched)

        print("Running temporal analysis...")
        self.ped_temporal = add_temporal_features(self.ped_data)
        self.hourly_profile = get_hourly_profile(self.ped_temporal)
        self.weekday_weekend = get_weekday_weekend_profile(self.ped_temporal)
        self.dimming_schedule = suggest_dimming_schedule(self.hourly_profile)

        print("Data ready.")


# ============================================================
# Query parsing — LLM-powered with regex fallback
# ============================================================

PARSE_SYSTEM_PROMPT = f"""\
You are a street lighting query parser. Extract design parameters from the user's \
natural language query and return ONLY a JSON object with these fields:

{{
  "location": string — the Melbourne park or location name exactly as the user \
stated it (e.g. "Fitzroy Gardens", "Royal Park"). Correct obvious typos but \
preserve the name. If no location mentioned, use "Melbourne CBD".
  "length": number — pathway length in metres. Default 200 if not specified.
  "width": number — pathway width in metres. Default 3.0 if not specified.
  "activity_level": "low" | "moderate" | "high" | "very_high" — based on described \
pedestrian traffic, time of day, or usage. Evening/night with people = high. \
Quiet/empty = low. Default "moderate".
  "location_type": "park_path" | "shared_path" | "public_space" | "residential" — \
infer from context. Parks/gardens = park_path. Roads/streets = shared_path. \
Plazas/squares = public_space. Suburbs = residential. Default "park_path".
}}

Return ONLY valid JSON. No markdown, no explanation, no extra text."""


def parse_query_llm(user_message: str) -> dict:
    """Use a fast LLM to extract structured design parameters from natural language."""
    resp = requests.post(
        f"{LM_STUDIO_BASE_URL}/chat/completions",
        json={
            "model": LM_STUDIO_PARSE_MODEL,
            "messages": [
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": PARSE_MAX_TOKENS,
            "temperature": FAST_LLM_TEMPERATURE,
        },
        timeout=FAST_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wraps the JSON
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)

    loc = parsed.get("location", "Melbourne CBD").strip()
    # Title-case for consistency (e.g. "fitzroy gardens" → "Fitzroy Gardens")
    if loc.islower():
        loc = loc.title()

    return {
        "location": loc,
        "length": float(parsed.get("length", 200)),
        "width": float(parsed.get("width", 3.0)),
        "activity_level": parsed.get("activity_level", "moderate"),
        "location_type": parsed.get("location_type", "park_path"),
    }


def parse_query_regex(user_message: str) -> dict:
    """Regex-based fallback parser. Fast, deterministic, no LLM needed."""
    msg = user_message.lower()

    length_match = re.search(r"(\d+)\s*(?:m\b|meters?|metres?)", msg)
    length = float(length_match.group(1)) if length_match else 200.0

    width_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\b|meters?|metres?)\s*wide", msg)
    width = float(width_match.group(1)) if width_match else 3.0

    location = "Melbourne CBD"
    for loc_name in KNOWN_LOCATIONS:
        if loc_name in msg:
            location = loc_name.title()
            break

    activity = "moderate"
    if any(w in msg for w in ["very busy", "extremely", "major", "transit hub"]):
        activity = "very_high"
    elif any(w in msg for w in ["busy", "heavy", "high", "peak", "crowded"]):
        activity = "high"
    elif any(w in msg for w in ["quiet", "low", "minimal", "empty", "sparse"]):
        activity = "low"

    loc_type = "park_path"
    if any(w in msg for w in ["road", "street", "intersection", "crossing"]):
        loc_type = "shared_path"
    elif any(w in msg for w in ["plaza", "square", "public space", "event"]):
        loc_type = "public_space"
    elif any(w in msg for w in ["residential", "suburb"]):
        loc_type = "residential"

    return {
        "location": location,
        "length": length,
        "width": width,
        "activity_level": activity,
        "location_type": loc_type,
    }


def parse_query(user_message: str) -> dict:
    """Parse a natural language query into design parameters.

    Uses LLM for accurate extraction (handles typos, synonyms, context).
    Falls back to regex if LLM is unavailable.
    """
    try:
        return parse_query_llm(user_message)
    except Exception as e:
        print(f"LLM parse failed ({e}), using regex fallback.")
        return parse_query_regex(user_message)


# ============================================================
# Spatial context
# ============================================================


def get_spatial_context(
    data: AppData,
    location: str,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict:
    """
    Look up spatial lighting context for a Melbourne location.

    Args:
        data: Loaded Melbourne Open Data.
        location: Human-readable location name.
        bounds: (lat_min, lat_max, lon_min, lon_max) from OSM boundary.
                If None, spatial context is unavailable.
    """
    if bounds is None:
        print(
            f"  No OSM boundary available for '{location}' — skipping spatial context."
        )
        return {
            "area_name": location,
            "num_streetlights": 0,
            "avg_lux_level": None,
            "avg_pedestrian_count": 0,
            "efficiency_breakdown": None,
        }

    return get_area_lighting_context(
        data.ped_matched,
        data.streetlight_data,
        lat_min=bounds[0],
        lat_max=bounds[1],
        lon_min=bounds[2],
        lon_max=bounds[3],
        area_name=location,
    )


# ============================================================
# Design orchestration
# ============================================================


def get_safety_context(location: str) -> dict | None:
    """Look up crime/safety risk for a location. Returns safety score dict or None."""
    try:
        from data.safety_analysis import get_lga_for_location, calculate_safety_score

        lga = get_lga_for_location(location)
        return calculate_safety_score(lga)
    except Exception as e:
        print(f"Safety analysis failed for '{location}': {e}")
        return None


def resolve_osm_pathway(location: str) -> dict | None:
    """Attempt to resolve real pathway geometry from OpenStreetMap.

    Returns the resolve_pathway result dict or None on failure.
    """
    try:
        from data.osm_loader import resolve_pathway

        return resolve_pathway(location)
    except Exception as e:
        print(f"OSM resolution failed for '{location}': {e}")
        return None


def run_design(
    data: AppData,
    query: str,
    *,
    location: str = None,
    pathway_length_m: float = None,
    pathway_width_m: float = None,
    activity_level: str = None,
    location_type: str = None,
):
    """
    Run the full deterministic design pipeline: parse → spatial → OSM → calc → dimming.

    Returns (design, calc_report, spatial, dimming_savings, full_context).
    """
    parsed = parse_query(query)
    location = location or parsed["location"]
    length = pathway_length_m or parsed["length"]
    width = pathway_width_m or parsed["width"]
    activity = activity_level or parsed["activity_level"]
    loc_type = location_type or parsed["location_type"]
    print(
        f"Parsed query parameters: location={location}, length={length}m, width={width}m, activity_level={activity}, location_type={loc_type}"
    )

    # --- OSM pathway resolution (must come first — provides bounds for spatial) ---
    osm_data = resolve_osm_pathway(location)
    osm_info = ""
    map_data = None
    osm_bounds = None
    # print osm_data for debugging
    print(f"OSM data for '{location}': {json.dumps(osm_data, indent=2)}")

    if osm_data:
        park_boundary = osm_data.get("park_boundary")
        if park_boundary:
            osm_bounds = bounds_from_osm_boundary(park_boundary)

        if osm_data.get("pathways"):
            sel_idx = osm_data.get("selected_pathway", 0) or 0
            selected_pw = osm_data["pathways"][sel_idx]
            osm_length = selected_pw["true_length_m"]

            # If user didn't specify length explicitly, use OSM length
            if pathway_length_m is None:
                if abs(osm_length - parsed["length"]) > 50:
                    osm_info = (
                        f"\nOSM PATHWAY DATA:\n"
                        f"  OSM measured length: {osm_length:.0f}m "
                        f"(user/default hint was {parsed['length']:.0f}m)\n"
                    )
                length = osm_length

            # Prepare map data for frontend
            map_data = {
                "pathway_geojson": selected_pw["geometry"],
                "park_boundary": park_boundary,
                "all_pathways": [pw["geometry"] for pw in osm_data["pathways"]],
                "intersections": selected_pw.get("intersections", []),
                "entry_points": selected_pw.get("entry_points", []),
            }

    if not osm_data:
        osm_info = (
            "\nOSM PATHWAY DATA: Not available — both Overpass and Nominatim failed.\n"
            "  Design is based on user-stated parameters only.\n"
            "  No real sensor data or existing streetlight data for this area.\n"
        )
    elif osm_bounds and not osm_data.get("pathways"):
        data_source = osm_data.get("data_source", "overpass")
        osm_info = (
            f"\nOSM PATHWAY DATA: Boundary resolved via {data_source} (no pathway geometry).\n"
            f"  Spatial context (sensors, streetlights) available.\n"
            f"  Light placement uses straight-line assumption.\n"
        )

    # --- Spatial context (uses OSM-derived bounds) ---
    spatial = get_spatial_context(data, location, bounds=osm_bounds)
    print(
        f"Spatial context for {location}: {spatial.get('num_streetlights', 'N/A')} streetlights, "
        f"avg lux {spatial.get('avg_lux_level', 'N/A')}, "
        f"avg pedestrian count {spatial.get('avg_pedestrian_count', 'N/A')}/hr"
    )

    # print spatial data for debugging
    print(f"Spatial data for '{location}': {json.dumps(spatial, indent=2)}")

    # --- Safety / crime risk assessment ---
    safety_context = get_safety_context(location)
    safety_info = ""
    safety_adjustment = 0

    if safety_context:
        safety_adjustment = safety_context.get("p_category_adjustment", 0)
        if safety_adjustment != 0:
            safety_info = (
                f"\nSAFETY RISK ASSESSMENT:\n"
                f"  LGA: {safety_context['lga_name']}\n"
                f"  Risk score: {safety_context['safety_risk_score']}/10 ({safety_context['risk_category']})\n"
                f"  P-category adjustment: {safety_adjustment} (negative = upgrade)\n"
                f"  {safety_context['recommendation']}\n"
            )

    avg_traffic = spatial.get("avg_pedestrian_count", 0)
    if avg_traffic == 0:
        avg_traffic = data.sensor_summary["avg_hourly_traffic"].median()

    # Pass geometry data to calc engine if available
    pw_geom = None
    pw_intersections = None
    pw_entries = None
    if map_data:
        pw_geom = map_data.get("pathway_geojson")
        pw_intersections = map_data.get("intersections")
        pw_entries = map_data.get("entry_points")

    design = design_lighting(
        location_name=location,
        pathway_length_m=length,
        pathway_width_m=width,
        location_type=loc_type,
        avg_pedestrian_count=avg_traffic,
        safety_adjustment=safety_adjustment,
        pathway_geometry=pw_geom,
        intersections=pw_intersections,
        entry_points=pw_entries,
    )
    calc_report = format_design_report(design)

    dimming_savings = estimate_dimming_savings(
        data.dimming_schedule, design.num_lights, design.led_wattage
    )

    # Build context string for LLM
    spatial_info = (
        f"\nSPATIAL DATA (from Melbourne Open Data):\n"
        f"  Area streetlights: {spatial.get('num_streetlights', 'N/A')}\n"
        f"  Avg existing lux: {spatial.get('avg_lux_level', 'N/A')}\n"
        f"  Avg pedestrian traffic: {avg_traffic:.0f}/hr\n"
    )
    if spatial.get("efficiency_breakdown"):
        spatial_info += (
            f"  Current efficiency: {json.dumps(spatial['efficiency_breakdown'])}\n"
        )

    dimming_info = (
        f"\nADAPTIVE DIMMING ANALYSIS (from temporal pedestrian data):\n"
        f"  Additional energy saving from dimming: {dimming_savings['saving_percent']:.1f}%\n"
        f"  Annual dimming saving: ${dimming_savings['annual_saving_cost_aud']:.2f}\n"
    )

    full_context = calc_report + spatial_info + osm_info + safety_info + dimming_info

    # Attach enriched data to spatial dict for API layer access
    spatial["map_data"] = map_data
    spatial["safety_context"] = safety_context
    spatial["retrieval_query"] = build_retrieval_query(parsed, design, safety_context)

    return design, calc_report, spatial, dimming_savings, full_context


def build_retrieval_query(parsed_query: dict, design, safety: dict | None) -> str:
    """Build a targeted retrieval query from design decisions, not raw user input."""
    parts = [
        f"P{design.p_category.replace('P', '')} category" if design.p_category else "",
        "pathway lighting",
        parsed_query.get("location_type", "park_path").replace("_", " "),
        f"{design.led_wattage}W LED luminaire",
        f"spacing {design.spacing_m}m",
        f"pedestrian traffic {parsed_query.get('activity_level', 'moderate')}",
    ]
    if safety and safety.get("risk_category") != "low":
        parts.append("safety risk crime prevention CPTED")
    return " ".join(p for p in parts if p)
