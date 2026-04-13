"""
Tests for crime/safety data integration and P-category adjustment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.safety_analysis import (
    load_crime_data,
    get_lga_for_location,
    calculate_safety_score,
    adjust_p_category,
)
from llm.calculation_engine import design_lighting


# ============================================================
# Crime data loading
# ============================================================

class TestCrimeDataLoading:
    def test_load_returns_dataframe(self):
        df = load_crime_data()
        assert not df.empty
        assert "lga_name" in df.columns
        assert "rate_per_100k" in df.columns

    def test_filters_relevant_offences(self):
        df = load_crime_data()
        valid = {"Assault", "Robbery", "Property Damage", "Stalking"}
        assert set(df["offence_category"].unique()).issubset(valid)


# ============================================================
# LGA lookup
# ============================================================

class TestLGALookup:
    def test_fitzroy_gardens(self):
        assert get_lga_for_location("Fitzroy Gardens") == "Melbourne"

    def test_edinburgh_gardens(self):
        assert get_lga_for_location("Edinburgh Gardens") == "Yarra"

    def test_unknown_defaults_to_melbourne(self):
        assert get_lga_for_location("Some Unknown Park") == "Melbourne"

    def test_case_insensitive(self):
        assert get_lga_for_location("FITZROY GARDENS") == "Melbourne"


# ============================================================
# Safety score calculation
# ============================================================

class TestSafetyScore:
    def test_score_range(self):
        result = calculate_safety_score("Melbourne")
        assert 0 <= result["safety_risk_score"] <= 10

    def test_returns_required_fields(self):
        result = calculate_safety_score("Melbourne")
        assert "risk_category" in result
        assert "p_category_adjustment" in result
        assert "recommendation" in result
        assert result["risk_category"] in ("low", "moderate", "high")

    def test_unknown_lga_returns_default(self):
        result = calculate_safety_score("NonexistentLGA")
        assert result["safety_risk_score"] == 5.0
        assert result["p_category_adjustment"] == 0

    def test_high_crime_area_has_negative_adjustment(self):
        """Yarra has higher crime rates, should get upgrade (negative adjustment)."""
        result = calculate_safety_score("Yarra")
        assert result["p_category_adjustment"] <= 0

    def test_low_crime_area_has_zero_adjustment(self):
        """Boroondara has lower crime rates, should have no upgrade."""
        result = calculate_safety_score("Boroondara")
        assert result["p_category_adjustment"] == 0


# ============================================================
# P-category adjustment
# ============================================================

class TestAdjustPCategory:
    def test_upgrade_one_level(self):
        assert adjust_p_category("P5", -1) == "P4"

    def test_upgrade_two_levels(self):
        assert adjust_p_category("P5", -2) == "P3"

    def test_clamp_at_p1(self):
        """Can't go below P1."""
        assert adjust_p_category("P1", -1) == "P1"
        assert adjust_p_category("P2", -3) == "P1"

    def test_clamp_at_p12(self):
        assert adjust_p_category("P12", 1) == "P12"

    def test_no_adjustment(self):
        assert adjust_p_category("P9", 0) == "P9"


# ============================================================
# Integration: safety adjustment in design_lighting
# ============================================================

class TestSafetyInDesignLighting:
    def test_safety_adjustment_changes_category(self):
        """Design with safety_adjustment=-1 should use a higher P-category."""
        d_base = design_lighting("Test", 200, 3.0, "moderate", "park_path")
        d_safe = design_lighting("Test", 200, 3.0, "moderate", "park_path", safety_adjustment=-1)

        base_num = int(d_base.p_category.replace("P", ""))
        safe_num = int(d_safe.p_category.replace("P", ""))
        assert safe_num == base_num - 1

    def test_safety_adjustment_clamped(self):
        """Very high activity with large adjustment should clamp at P1."""
        d = design_lighting("Test", 200, 3.0, "very_high", "shared_path", safety_adjustment=-5)
        assert d.p_category == "P1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
