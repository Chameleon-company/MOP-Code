"""
Unit tests for the AS/NZS 1158 lighting calculation engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import pytest
from llm.calculation_engine import (
    design_lighting,
    select_p_category,
    select_led_spec,
    calculate_spacing,
    select_pole_height,
    P_CATEGORIES,
    OPERATING_HOURS_PER_YEAR,
    ELECTRICITY_RATE_PER_KWH,
    CARBON_FACTOR_VIC_SCOPE2_3,
)


# ============================================================
# P-Category Selection
# ============================================================

class TestCategorySelection:
    def test_park_path_low(self):
        assert select_p_category("low", "park_path") == "P10"

    def test_park_path_moderate(self):
        assert select_p_category("moderate", "park_path") == "P9"

    def test_park_path_high(self):
        assert select_p_category("high", "park_path") == "P3"

    def test_park_path_very_high(self):
        assert select_p_category("very_high", "park_path") == "P2"

    def test_shared_path_high(self):
        assert select_p_category("high", "shared_path") == "P2"

    def test_residential_low(self):
        assert select_p_category("low", "residential") == "P8"

    def test_all_categories_exist(self):
        for level in ["low", "moderate", "high", "very_high"]:
            for loc_type in ["park_path", "shared_path", "public_space", "residential"]:
                cat = select_p_category(level, loc_type)
                assert cat in P_CATEGORIES, f"{cat} not in P_CATEGORIES"


# ============================================================
# Spacing and Pole Height
# ============================================================

class TestSpacingAndPoleHeight:
    def test_spacing_within_range(self):
        """Spacing should be 3-5x pole height per AS/NZS 1158."""
        for cat_id in P_CATEGORIES:
            height = select_pole_height(cat_id, 3.0)
            spacing = calculate_spacing(height, cat_id)
            ratio = spacing / height
            assert 3.0 <= ratio <= 5.0, f"{cat_id}: ratio {ratio} outside 3-5x range"

    def test_higher_categories_have_closer_spacing(self):
        """P1 should have closer spacing than P10 for same pole height."""
        sp_p1 = calculate_spacing(5.0, "P1")
        sp_p10 = calculate_spacing(5.0, "P10")
        assert sp_p1 < sp_p10

    def test_pole_height_minimum(self):
        for cat_id in P_CATEGORIES:
            height = select_pole_height(cat_id, 2.0)
            assert height >= 3.5, f"{cat_id}: pole height {height}m too low"


# ============================================================
# Light Count (Fence-Post Formula)
# ============================================================

class TestLightCount:
    def test_200m_path_p9(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        # P9: pole 4m, spacing 16m -> floor(200/16)+1 = 13
        assert d.num_lights == 13

    def test_200m_path_p3(self):
        d = design_lighting("Test", 200, 3.0, "high", "park_path")
        # P3: pole 5m, spacing 17.5m -> floor(200/17.5)+1 = 12
        assert d.num_lights == 12

    def test_minimum_2_lights(self):
        d = design_lighting("Short", 5, 2.0, "low", "park_path")
        assert d.num_lights >= 2

    def test_longer_path_more_lights(self):
        d1 = design_lighting("Short", 100, 3.0, "moderate", "park_path")
        d2 = design_lighting("Long", 500, 3.0, "moderate", "park_path")
        assert d2.num_lights > d1.num_lights


# ============================================================
# Energy Calculations
# ============================================================

class TestEnergyCalculations:
    def test_energy_formula(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        expected_kwh = (d.num_lights * d.led_wattage * OPERATING_HOURS_PER_YEAR) / 1000
        assert abs(d.annual_energy_kwh - expected_kwh) < 0.01

    def test_cost_formula(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        expected_cost = d.annual_energy_kwh * ELECTRICITY_RATE_PER_KWH
        assert abs(d.annual_energy_cost_aud - expected_cost) < 0.01

    def test_co2_formula(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        expected_co2 = d.annual_energy_kwh * CARBON_FACTOR_VIC_SCOPE2_3
        assert abs(d.annual_co2_kg - expected_co2) < 0.01

    def test_led_cheaper_than_hps(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        assert d.annual_energy_cost_aud < d.hps_annual_cost_aud

    def test_energy_saving_positive(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        assert d.energy_saving_percent > 0

    def test_energy_saving_range(self):
        """LED should save 50-70% vs HPS."""
        d = design_lighting("Test", 200, 3.0, "high", "park_path")
        assert 50 <= d.energy_saving_percent <= 75


# ============================================================
# Activity Level from Pedestrian Count
# ============================================================

class TestActivityLevelOverride:
    def test_low_traffic(self):
        d = design_lighting("Test", 200, 3.0, avg_pedestrian_count=30)
        assert d.activity_level == "low"

    def test_moderate_traffic(self):
        d = design_lighting("Test", 200, 3.0, avg_pedestrian_count=150)
        assert d.activity_level == "moderate"

    def test_high_traffic(self):
        d = design_lighting("Test", 200, 3.0, avg_pedestrian_count=500)
        assert d.activity_level == "high"

    def test_very_high_traffic(self):
        d = design_lighting("Test", 200, 3.0, avg_pedestrian_count=1500)
        assert d.activity_level == "very_high"


# ============================================================
# Payback Period
# ============================================================

class TestPayback:
    def test_payback_positive(self):
        d = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        assert d.payback_years > 0

    def test_payback_reasonable(self):
        """Payback should be within 3-30 years for realistic scenarios."""
        d = design_lighting("Test", 200, 3.0, "high", "park_path")
        assert 3 <= d.payback_years <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
