"""
Tests for the output validation layer (LLM vs calculation engine cross-check).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.validator import (
    extract_numbers_from_report,
    validate_report,
    append_verification_footer,
)


# ============================================================
# Number extraction
# ============================================================

SAMPLE_REPORT = """
## Design Specification

This design proposes **12 lights** along the 200m pathway with a spacing of 17.5m.
The system uses P3 category luminaires rated at 60W each.

### Energy & Cost
- Annual energy consumption: 3,024 kWh
- Annual energy cost: $604.80/year
- CO2 emissions: 3,265.9 kg CO2-e per year
"""

SAMPLE_DESIGN = {
    "num_lights": 12,
    "spacing_m": 17.5,
    "p_category": "P3",
    "annual_energy_cost_aud": 604.80,
    "annual_energy_kwh": 3024.0,
    "annual_co2_kg": 3265.9,
    "led_wattage": 60,
    "total_capital_cost_aud": 54000,
    "energy_saving_vs_hps_percent": 65.7,
    "co2_saving_vs_hps_kg": 6200.0,
}


class TestExtractNumbers:
    def test_extracts_light_count(self):
        result = extract_numbers_from_report(SAMPLE_REPORT)
        assert result["light_count"] == 12

    def test_extracts_spacing(self):
        result = extract_numbers_from_report(SAMPLE_REPORT)
        assert result["spacing_m"] == 17.5

    def test_extracts_annual_cost(self):
        result = extract_numbers_from_report(SAMPLE_REPORT)
        assert result["annual_energy_cost"] == 604.80

    def test_extracts_energy_kwh(self):
        result = extract_numbers_from_report(SAMPLE_REPORT)
        assert result["annual_energy_kwh"] == 3024.0

    def test_extracts_p_category(self):
        result = extract_numbers_from_report(SAMPLE_REPORT)
        assert result["p_category"] == 3

    def test_handles_missing_values(self):
        result = extract_numbers_from_report("No numbers here.")
        assert all(v is None for v in result.values())


# ============================================================
# Validation
# ============================================================

class TestValidation:
    def test_matching_numbers_pass(self):
        result = validate_report(SAMPLE_REPORT, SAMPLE_DESIGN)
        assert result["valid"] is True
        assert len(result["mismatches"]) == 0
        assert result["needs_regeneration"] is False

    def test_wrong_light_count_flagged(self):
        wrong_report = SAMPLE_REPORT.replace("12 lights", "15 lights")
        result = validate_report(wrong_report, SAMPLE_DESIGN)
        assert result["valid"] is False
        assert any(m["field"] == "light_count" for m in result["mismatches"])

    def test_major_mismatch_needs_regeneration(self):
        wrong_report = SAMPLE_REPORT.replace("12 lights", "25 lights")
        result = validate_report(wrong_report, SAMPLE_DESIGN)
        assert result["needs_regeneration"] is True

    def test_minor_rounding_auto_corrected(self):
        """A 1-off light count should be auto-corrected."""
        close_report = SAMPLE_REPORT.replace("12 lights", "13 lights")
        result = validate_report(close_report, SAMPLE_DESIGN)
        minor = [m for m in result["mismatches"] if m["severity"] == "minor"]
        assert len(minor) >= 1
        # The corrected report should contain the right number
        assert "12 lights" in result["corrected_report"]

    def test_tolerance_parameter(self):
        """With large tolerance, small differences should pass."""
        slightly_off = SAMPLE_REPORT.replace("$604.80/year", "$610.00/year")
        result = validate_report(slightly_off, SAMPLE_DESIGN, tolerance=0.10)
        assert result["valid"] is True


# ============================================================
# Verification footer
# ============================================================

class TestVerificationFooter:
    def test_footer_appended(self):
        result = append_verification_footer("Some report text", SAMPLE_DESIGN)
        assert "Calculation Verification" in result
        assert "P3" in result
        assert "12" in result

    def test_footer_is_markdown_table(self):
        result = append_verification_footer("Report", SAMPLE_DESIGN)
        assert "| Parameter | Value |" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
