"""
Melbourne Open Data loader.

Fetches real pedestrian counting data and streetlight location/lux data
from the City of Melbourne Open Data Portal API.
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / "cache"


def fetch_melbourne_csv(dataset_name: str, limit: int = -1, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch a dataset from Melbourne Open Data Portal.

    Args:
        dataset_name: The dataset identifier on the portal.
        limit: Max rows to fetch. -1 for all (paginated).
        use_cache: If True, cache to CSV locally after first fetch.

    Returns:
        DataFrame with the dataset.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{dataset_name}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached {dataset_name}...")
        return pd.read_csv(cache_file)

    base_url = f"https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/{dataset_name}/exports/csv"

    if limit > 0 and limit <= 500:
        # Single request
        params = {"limit": limit, "format": "csv"}
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.content.decode("utf-8")), sep=";")
    else:
        # Paginate to get all rows (API max 500 per request)
        all_dfs = []
        offset = 0
        page_size = 500
        while True:
            fetch_size = min(page_size, limit - offset) if limit > 0 else page_size
            params = {"limit": fetch_size, "offset": offset, "format": "csv"}
            resp = requests.get(base_url, params=params)
            resp.raise_for_status()
            chunk = pd.read_csv(StringIO(resp.content.decode("utf-8")), sep=";")
            if chunk.empty:
                break
            all_dfs.append(chunk)
            print(f"  Fetched {len(chunk)} rows (offset {offset})...")
            offset += page_size
            if limit > 0 and offset >= limit:
                break
            if len(chunk) < page_size:
                break
            # Melbourne API has a 10,000 row export limit
            if offset >= 9500:
                break
        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Cache locally
    if use_cache and not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"Cached {len(df)} rows to {cache_file}")

    return df


def load_pedestrian_data(limit: int = 5000) -> pd.DataFrame:
    """
    Load Melbourne pedestrian counting data.

    Returns DataFrame with columns:
        sensor_name, location, sensing_date, hourday,
        direction_1, direction_2, pedestriancount, Latitude, Longitude
    """
    df = fetch_melbourne_csv(
        "pedestrian-counting-system-monthly-counts-per-hour",
        limit=limit,
    )

    # Parse coordinates from location string "lat, lon"
    if "location" in df.columns:
        coords = df["location"].dropna().str.split(",", expand=True)
        df.loc[coords.index, "Latitude"] = coords[0].str.strip().astype(float)
        df.loc[coords.index, "Longitude"] = coords[1].str.strip().astype(float)

    # Ensure numeric columns
    for col in ["direction_1", "direction_2", "pedestriancount", "hourday"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Total foot traffic
    if "direction_1" in df.columns and "direction_2" in df.columns:
        df["total_traffic"] = df["direction_1"].fillna(0) + df["direction_2"].fillna(0)

    print(f"Pedestrian data: {len(df)} rows, {df['sensor_name'].nunique()} unique sensors")
    return df


def load_streetlight_data(limit: int = -1) -> pd.DataFrame:
    """
    Load Melbourne council-owned streetlight data with lux levels.

    Returns DataFrame with columns:
        geo_point_2d, label (lux level), Latitude, Longitude, and original cols
    """
    df = fetch_melbourne_csv(
        "street-lights-with-emitted-lux-level-council-owned-lights-only",
        limit=limit,
    )

    # Parse coordinates
    if "geo_point_2d" in df.columns:
        coords = df["geo_point_2d"].dropna().str.split(",", expand=True)
        df.loc[coords.index, "Latitude"] = coords[0].str.strip().astype(float)
        df.loc[coords.index, "Longitude"] = coords[1].str.strip().astype(float)

    # Lux level is in the 'label' column
    if "label" in df.columns:
        df["lux_level"] = pd.to_numeric(df["label"], errors="coerce")

    print(f"Streetlight data: {len(df)} rows")
    return df


def get_sensor_summary(ped_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pedestrian data per sensor: location, avg/max/total traffic.
    Useful for understanding which areas have high pedestrian activity.
    """
    required_cols = ["sensor_name", "Latitude", "Longitude", "pedestriancount"]
    missing = [c for c in required_cols if c not in ped_df.columns]
    if missing:
        raise ValueError(f"Pedestrian data missing required columns: {missing}")

    summary = ped_df.groupby("sensor_name").agg(
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
        avg_hourly_traffic=("pedestriancount", "mean"),
        max_hourly_traffic=("pedestriancount", "max"),
        total_records=("pedestriancount", "count"),
    ).reset_index()

    summary = summary.sort_values("avg_hourly_traffic", ascending=False)
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("Loading Melbourne Pedestrian Data")
    print("=" * 60)
    ped = load_pedestrian_data(limit=2000)
    print(ped.head())
    print()

    print("=" * 60)
    print("Loading Melbourne Streetlight Data")
    print("=" * 60)
    lights = load_streetlight_data()
    print(lights.head())
    print()

    print("=" * 60)
    print("Sensor Summary (Top 10 busiest)")
    print("=" * 60)
    summary = get_sensor_summary(ped)
    print(summary.head(10).to_string())
