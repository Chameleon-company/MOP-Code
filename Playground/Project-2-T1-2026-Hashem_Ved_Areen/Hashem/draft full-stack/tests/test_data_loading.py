"""
Tests for Melbourne data loading and spatial analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np


class TestDataLoading:
    """Test the Melbourne Open Data loading functions."""

    def test_pedestrian_data_has_required_columns(self):
        from data.load_melbourne_data import load_pedestrian_data
        df = load_pedestrian_data(limit=500)
        required = ["sensor_name", "pedestriancount", "Latitude", "Longitude"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_pedestrian_data_coordinates_valid(self):
        from data.load_melbourne_data import load_pedestrian_data
        df = load_pedestrian_data(limit=500)
        valid = df.dropna(subset=["Latitude", "Longitude"])
        assert len(valid) > 0, "No valid coordinates"
        # Melbourne bounding box
        assert valid["Latitude"].between(-38.5, -37.5).all(), "Latitude out of Melbourne range"
        assert valid["Longitude"].between(144.5, 145.5).all(), "Longitude out of Melbourne range"

    def test_streetlight_data_has_lux(self):
        from data.load_melbourne_data import load_streetlight_data
        df = load_streetlight_data(limit=500)
        assert "lux_level" in df.columns
        assert df["lux_level"].notna().sum() > 0

    def test_streetlight_lux_values_reasonable(self):
        from data.load_melbourne_data import load_streetlight_data
        df = load_streetlight_data(limit=500)
        lux = df["lux_level"].dropna()
        assert lux.min() >= 0, "Negative lux values"
        assert lux.max() < 200, "Unreasonably high lux values"

    def test_sensor_summary(self):
        from data.load_melbourne_data import load_pedestrian_data, get_sensor_summary
        df = load_pedestrian_data(limit=500)
        summary = get_sensor_summary(df)
        assert "sensor_name" in summary.columns
        assert "avg_hourly_traffic" in summary.columns
        assert len(summary) > 0


class TestSpatialAnalysis:
    """Test spatial analysis functions."""

    def test_haversine_distances_reasonable(self):
        from data.load_melbourne_data import load_pedestrian_data, load_streetlight_data
        from data.spatial_analysis import match_sensors_to_lights

        ped = load_pedestrian_data(limit=500)
        lights = load_streetlight_data(limit=500)
        matched = match_sensors_to_lights(ped, lights)

        distances = matched["distance_to_light_m"].dropna()
        assert distances.min() > 0, "Zero distance — likely a bug"
        assert distances.max() < 10000, "Distance > 10km — wrong scale"
        assert distances.mean() < 2000, "Average distance > 2km — seems too large"

    def test_efficiency_categories_valid(self):
        from data.load_melbourne_data import load_pedestrian_data, load_streetlight_data
        from data.spatial_analysis import match_sensors_to_lights, analyze_lighting_efficiency

        ped = load_pedestrian_data(limit=500)
        lights = load_streetlight_data(limit=500)
        ped = match_sensors_to_lights(ped, lights)
        ped = analyze_lighting_efficiency(ped)

        valid_cats = {"Efficient", "Adequate", "Underlit", "Overlit", "No Nearby Light"}
        actual_cats = set(ped["efficiency"].unique())
        assert actual_cats.issubset(valid_cats), f"Unexpected categories: {actual_cats - valid_cats}"


class TestTemporalAnalysis:
    """Test temporal analysis functions."""

    def test_hourly_profile_24_hours(self):
        from data.load_melbourne_data import load_pedestrian_data
        from data.temporal_analysis import add_temporal_features, get_hourly_profile

        ped = load_pedestrian_data(limit=2000)
        ped = add_temporal_features(ped)
        hourly = get_hourly_profile(ped)

        assert len(hourly) > 0, "Empty hourly profile"
        assert hourly["hour"].min() >= 0
        assert hourly["hour"].max() <= 23

    def test_dimming_schedule_valid(self):
        from data.load_melbourne_data import load_pedestrian_data
        from data.temporal_analysis import (
            add_temporal_features, get_hourly_profile,
            suggest_dimming_schedule,
        )

        ped = load_pedestrian_data(limit=2000)
        ped = add_temporal_features(ped)
        hourly = get_hourly_profile(ped)
        schedule = suggest_dimming_schedule(hourly)

        assert len(schedule) > 0
        for entry in schedule:
            assert entry["suggested_output_pct"] in {40, 60, 80, 100}
            assert 0 <= entry["hour"] <= 23

    def test_dimming_savings_positive(self):
        from data.load_melbourne_data import load_pedestrian_data
        from data.temporal_analysis import (
            add_temporal_features, get_hourly_profile,
            suggest_dimming_schedule, estimate_dimming_savings,
        )

        ped = load_pedestrian_data(limit=2000)
        ped = add_temporal_features(ped)
        hourly = get_hourly_profile(ped)
        schedule = suggest_dimming_schedule(hourly)
        savings = estimate_dimming_savings(schedule, 13, 30)

        assert savings["saving_percent"] > 0, "No dimming savings"
        assert savings["saving_percent"] < 80, "Unrealistic savings > 80%"
        assert savings["annual_saving_cost_aud"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
