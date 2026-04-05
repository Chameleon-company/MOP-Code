"""
Temporal analysis of pedestrian activity patterns.

Analyzes hourly, daily, and seasonal pedestrian traffic patterns
to inform adaptive dimming schedules and peak lighting requirements.
"""

import pandas as pd
import numpy as np


def add_temporal_features(ped_df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal columns: hour, day_of_week, day_type, time_period."""
    df = ped_df.copy()

    if "sensing_date" in df.columns:
        df["date"] = pd.to_datetime(df["sensing_date"], errors="coerce")
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun
        df["day_type"] = df["day_of_week"].apply(
            lambda x: "Weekend" if x >= 5 else "Weekday"
        )
        df["month"] = df["date"].dt.month

    if "hourday" in df.columns:
        df["hour"] = pd.to_numeric(df["hourday"], errors="coerce")
        df["time_period"] = df["hour"].apply(classify_time_period)

    return df


def classify_time_period(hour: float) -> str:
    """Classify hour into lighting-relevant time periods."""
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if 5 <= hour < 7:
        return "Dawn"
    elif 7 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening Peak"
    elif 21 <= hour < 24:
        return "Night"
    else:  # 0-4
        return "Late Night"


def get_hourly_profile(ped_df: pd.DataFrame) -> pd.DataFrame:
    """Get average pedestrian count by hour of day."""
    if "hour" not in ped_df.columns:
        ped_df = add_temporal_features(ped_df)

    hourly = ped_df.groupby("hour")["pedestriancount"].agg(
        ["mean", "median", "max", "count"]
    ).reset_index()
    hourly.columns = ["hour", "avg_traffic", "median_traffic", "peak_traffic", "sample_count"]
    return hourly


def get_weekday_weekend_profile(ped_df: pd.DataFrame) -> pd.DataFrame:
    """Get average pedestrian count by hour, split by weekday/weekend."""
    if "hour" not in ped_df.columns or "day_type" not in ped_df.columns:
        ped_df = add_temporal_features(ped_df)

    profile = ped_df.groupby(["hour", "day_type"])["pedestriancount"].mean().reset_index()
    profile.columns = ["hour", "day_type", "avg_traffic"]
    return profile


def suggest_dimming_schedule(hourly_profile: pd.DataFrame) -> list[dict]:
    """
    Suggest an adaptive dimming schedule based on hourly traffic patterns.

    Returns a list of dicts: [{hour, traffic_level, suggested_output_pct, reason}]
    """
    if hourly_profile.empty:
        return []

    max_traffic = hourly_profile["avg_traffic"].max()
    schedule = []

    for _, row in hourly_profile.iterrows():
        hour = int(row["hour"])
        traffic = row["avg_traffic"]
        ratio = traffic / max_traffic if max_traffic > 0 else 0

        if ratio >= 0.6:
            output = 100
            reason = "Peak/high activity — full illuminance"
        elif ratio >= 0.3:
            output = 80
            reason = "Moderate activity — reduced one P-category step"
        elif ratio >= 0.1:
            output = 60
            reason = "Low activity — reduced two P-category steps"
        else:
            output = 40
            reason = "Very low activity — minimum safe level"

        schedule.append({
            "hour": hour,
            "avg_traffic": round(traffic, 1),
            "traffic_ratio": round(ratio, 2),
            "suggested_output_pct": output,
            "reason": reason,
        })

    return schedule


def format_dimming_schedule(schedule: list[dict]) -> str:
    """Format dimming schedule as readable text."""
    if not schedule:
        return "No data available for dimming schedule."

    lines = ["Hour | Traffic | Output | Reason", "------|---------|--------|-------"]
    for entry in schedule:
        lines.append(
            f"{entry['hour']:02d}:00 | {entry['avg_traffic']:>7.1f} | "
            f"{entry['suggested_output_pct']:>5d}% | {entry['reason']}"
        )
    return "\n".join(lines)


def estimate_dimming_savings(
    schedule: list[dict],
    num_lights: int,
    led_wattage: int,
    electricity_rate: float = 0.20,
) -> dict:
    """
    Estimate annual energy and cost savings from adaptive dimming.
    Assumes each hour of the day occurs 365 times per year.
    """
    full_power_kwh = num_lights * led_wattage * 365 / 1000  # per hour-slot per year
    total_full_kwh = 0
    total_dimmed_kwh = 0

    # Only count hours when lights are on (roughly 5pm-7am = sunset to sunrise avg)
    lighting_hours = set(range(0, 7)) | set(range(17, 24))

    for entry in schedule:
        if entry["hour"] in lighting_hours:
            total_full_kwh += full_power_kwh
            dimmed_kwh = full_power_kwh * (entry["suggested_output_pct"] / 100)
            total_dimmed_kwh += dimmed_kwh

    saving_kwh = total_full_kwh - total_dimmed_kwh
    saving_cost = saving_kwh * electricity_rate
    saving_pct = (saving_kwh / total_full_kwh * 100) if total_full_kwh > 0 else 0

    return {
        "annual_full_power_kwh": round(total_full_kwh, 1),
        "annual_dimmed_kwh": round(total_dimmed_kwh, 1),
        "annual_saving_kwh": round(saving_kwh, 1),
        "annual_saving_cost_aud": round(saving_cost, 2),
        "saving_percent": round(saving_pct, 1),
    }


if __name__ == "__main__":
    from load_melbourne_data import load_pedestrian_data

    print("Loading data...")
    ped = load_pedestrian_data(limit=2000)
    ped = add_temporal_features(ped)

    print("\nHourly Profile:")
    hourly = get_hourly_profile(ped)
    print(hourly.to_string(index=False))

    print("\nSuggested Dimming Schedule:")
    schedule = suggest_dimming_schedule(hourly)
    print(format_dimming_schedule(schedule))

    print("\nDimming Savings (13 x 30W LED):")
    savings = estimate_dimming_savings(schedule, num_lights=13, led_wattage=30)
    for k, v in savings.items():
        print(f"  {k}: {v}")
