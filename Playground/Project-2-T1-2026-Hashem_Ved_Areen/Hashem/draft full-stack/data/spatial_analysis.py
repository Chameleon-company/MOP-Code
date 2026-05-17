"""
Spatial analysis module for street lighting design.

Matches pedestrian sensors to nearest streetlights, analyzes coverage,
identifies underlit/overlit areas, and provides spatial context for
the calculation engine and LLM.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def match_sensors_to_lights(
    ped_df: pd.DataFrame,
    light_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match each pedestrian sensor to its nearest streetlight using k-NN.

    Adds columns to ped_df:
        - nearest_light_idx: index in light_df
        - distance_to_light_m: approximate distance in metres
        - nearest_light_lux: lux level of nearest light

    Note: Distance is approximate (uses lat/lon directly, not haversine).
    At Melbourne's latitude, 1 degree lat ≈ 111km, 1 degree lon ≈ 82km.
    """
    ped_clean = ped_df.dropna(subset=["Latitude", "Longitude"]).copy()
    light_clean = light_df.dropna(subset=["Latitude", "Longitude"]).copy()

    if ped_clean.empty or light_clean.empty:
        print("Warning: No valid coordinates for matching.")
        return ped_df

    ped_coords = ped_clean[["Latitude", "Longitude"]].to_numpy()
    light_coords = light_clean[["Latitude", "Longitude"]].to_numpy()

    # Use haversine metric for geographically correct nearest-neighbor matching
    # ball_tree with haversine expects coordinates in radians
    EARTH_RADIUS_M = 6_371_000
    ped_rad = np.radians(ped_coords)
    light_rad = np.radians(light_coords)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric="haversine").fit(light_rad)
    distances, indices = nbrs.kneighbors(ped_rad)

    ped_clean["nearest_light_idx"] = indices.flatten()
    # haversine returns distance in radians; multiply by Earth radius for metres
    ped_clean["distance_to_light_m"] = distances.flatten() * EARTH_RADIUS_M

    # Attach lux level
    if "lux_level" in light_clean.columns:
        ped_clean["nearest_light_lux"] = ped_clean["nearest_light_idx"].apply(
            lambda i: light_clean.iloc[i]["lux_level"] if i < len(light_clean) else np.nan
        )

    # Copy results back to original dataframe
    for col in ["nearest_light_idx", "distance_to_light_m", "nearest_light_lux"]:
        if col in ped_clean.columns:
            ped_df.loc[ped_clean.index, col] = ped_clean[col]

    print(f"Matched {len(ped_clean)} sensors to nearest streetlights.")
    print(f"  Avg distance to nearest light: {ped_clean['distance_to_light_m'].mean():.1f}m")
    print(f"  Max distance to nearest light: {ped_clean['distance_to_light_m'].max():.1f}m")

    return ped_df


def analyze_lighting_efficiency(
    ped_df: pd.DataFrame,
    max_match_distance_m: float = 200.0,
) -> pd.DataFrame:
    """
    Classify each sensor location as Efficient, Underlit, or Overlit
    based on pedestrian traffic vs lighting levels.

    Thresholds aligned with AS/NZS 1158 P-categories:
        - High traffic (>=300/hr) needs P3+ (>=7 lux): "high-activity" zones
        - Moderate traffic (50-300/hr) needs P9+ (>=2 lux): park path standard
        - Low traffic (<50/hr): P10 (>=1 lux) is sufficient

    Only considers matches within max_match_distance_m (default 50m).
    Matches beyond this distance are classified as "No Nearby Light".
    """
    traffic_col = "pedestriancount" if "pedestriancount" in ped_df.columns else "total_traffic"
    lux_col = "nearest_light_lux"
    dist_col = "distance_to_light_m"

    if lux_col not in ped_df.columns or traffic_col not in ped_df.columns:
        print("Warning: Missing columns for efficiency analysis.")
        return ped_df

    # Mark matches beyond max distance as unreliable
    too_far = ped_df.get(dist_col, pd.Series(dtype=float)) > max_match_distance_m

    # Thresholds aligned with AS/NZS 1158:
    #   High traffic (>=300/hr) -> needs P3 level (7 lux avg)
    #   Moderate traffic (50-300) -> needs P9 level (2 lux avg)
    #   Low traffic (<50) -> P10 (1 lux) sufficient; >14 lux is wasteful (exceeds P1)
    conditions = [
        too_far,
        (ped_df[traffic_col] >= 300) & (ped_df[lux_col] >= 7),
        (ped_df[traffic_col] >= 300) & (ped_df[lux_col] < 7),
        (ped_df[traffic_col] >= 50) & (ped_df[traffic_col] < 300) & (ped_df[lux_col] >= 2),
        (ped_df[traffic_col] >= 50) & (ped_df[traffic_col] < 300) & (ped_df[lux_col] < 2),
        (ped_df[traffic_col] < 50) & (ped_df[lux_col] > 14),
    ]
    choices = ["No Nearby Light", "Efficient", "Underlit", "Adequate", "Underlit", "Overlit"]
    ped_df["efficiency"] = np.select(conditions, choices, default="Adequate")

    counts = ped_df["efficiency"].value_counts()
    print("\nLighting Efficiency Distribution:")
    for cat, count in counts.items():
        pct = count / len(ped_df) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    return ped_df


def suggest_adaptive_lux(foot_traffic: float) -> float:
    """
    Suggest appropriate lux level based on pedestrian traffic.
    Based on AS/NZS 1158 P-category thresholds.
    """
    if foot_traffic < 50:
        return 1.0  # P10 level
    elif foot_traffic < 300:
        return 2.0  # P9 level
    elif foot_traffic < 1000:
        return 7.0  # P3 level
    elif foot_traffic < 1000:
        return 10.0  # P2 level
    else:
        return 14.0  # P1 level


def get_area_lighting_context(
    ped_df: pd.DataFrame,
    light_df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    area_name: str = "Target Area",
) -> dict:
    """
    Get spatial lighting context for a specific geographic area.
    Returns a summary dict for the LLM to use.
    """
    # Filter to area
    ped_area = ped_df[
        (ped_df["Latitude"] >= lat_min) & (ped_df["Latitude"] <= lat_max) &
        (ped_df["Longitude"] >= lon_min) & (ped_df["Longitude"] <= lon_max)
    ]
    light_area = light_df[
        (light_df["Latitude"] >= lat_min) & (light_df["Latitude"] <= lat_max) &
        (light_df["Longitude"] >= lon_min) & (light_df["Longitude"] <= lon_max)
    ]

    context = {
        "area_name": area_name,
        "num_pedestrian_sensors": len(ped_area["sensor_name"].unique()) if "sensor_name" in ped_area.columns else 0,
        "num_streetlights": len(light_area),
        "avg_pedestrian_count": round(ped_area["pedestriancount"].mean(), 1) if not ped_area.empty else 0,
        "max_pedestrian_count": int(ped_area["pedestriancount"].max()) if not ped_area.empty else 0,
        "avg_lux_level": round(light_area["lux_level"].mean(), 1) if "lux_level" in light_area.columns and not light_area.empty else 0,
        "min_lux_level": round(light_area["lux_level"].min(), 1) if "lux_level" in light_area.columns and not light_area.empty else 0,
        "max_lux_level": round(light_area["lux_level"].max(), 1) if "lux_level" in light_area.columns and not light_area.empty else 0,
    }

    # Efficiency breakdown if available
    if "efficiency" in ped_area.columns and not ped_area.empty:
        eff_counts = ped_area["efficiency"].value_counts().to_dict()
        context["efficiency_breakdown"] = eff_counts

    return context


if __name__ == "__main__":
    from load_melbourne_data import load_pedestrian_data, load_streetlight_data

    print("Loading data...")
    ped = load_pedestrian_data(limit=2000)
    lights = load_streetlight_data(limit=5000)

    print("\nMatching sensors to streetlights...")
    ped = match_sensors_to_lights(ped, lights)

    print("\nAnalyzing efficiency...")
    ped = analyze_lighting_efficiency(ped)

    # Get context for Melbourne CBD area
    print("\nCBD area lighting context:")
    cbd_context = get_area_lighting_context(
        ped, lights,
        lat_min=-37.825, lat_max=-37.810,
        lon_min=144.955, lon_max=144.975,
        area_name="Melbourne CBD",
    )
    for k, v in cbd_context.items():
        print(f"  {k}: {v}")
