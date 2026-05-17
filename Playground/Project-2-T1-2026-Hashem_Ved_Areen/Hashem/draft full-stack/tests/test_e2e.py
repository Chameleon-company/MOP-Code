"""
End-to-end integration tests.
These test the full pipeline from query to response.
They do NOT require LM Studio — they test the deterministic pipeline only.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# Mock AppData for pipeline tests (no live Melbourne API needed)
# ============================================================

def _make_mock_appdata():
    """Create a minimal AppData-like mock for pipeline testing."""
    import pandas as pd
    import numpy as np

    mock = MagicMock()

    # Minimal sensor summary
    mock.sensor_summary = pd.DataFrame({
        "sensor_name": ["Sensor1", "Sensor2"],
        "Latitude": [-37.813, -37.815],
        "Longitude": [144.963, 144.965],
        "avg_hourly_traffic": [120.0, 350.0],
        "max_hourly_traffic": [500, 1200],
        "total_records": [1000, 2000],
    })

    # Minimal dimming schedule
    mock.dimming_schedule = [
        {"hour": h, "avg_traffic": 100, "traffic_ratio": 0.5, "suggested_output_pct": 80, "reason": "test"}
        for h in range(24)
    ]

    # Mock spatial analysis results
    mock.ped_matched = pd.DataFrame({
        "sensor_name": ["Sensor1"],
        "Latitude": [-37.813],
        "Longitude": [144.963],
        "pedestriancount": [200],
        "nearest_light_lux": [5.0],
        "distance_to_light_m": [30.0],
        "efficiency": ["Adequate"],
    })

    mock.streetlight_data = pd.DataFrame({
        "Latitude": [-37.813, -37.814],
        "Longitude": [144.963, 144.964],
        "lux_level": [5.0, 8.0],
    })

    return mock


# ============================================================
# E2E: New design request (deterministic, no LLM)
# ============================================================

class TestE2EDesign:
    def test_full_pipeline_fitzroy_gardens(self):
        """Run the full pipeline for Fitzroy Gardens (mocked data)."""
        from core.pipeline import run_design

        mock_data = _make_mock_appdata()

        # Mock OSM to avoid network call
        fixture = json.loads((FIXTURES_DIR / "fitzroy_gardens_resolved.json").read_text())
        with patch("core.pipeline.resolve_osm_pathway", return_value=fixture):
            design, calc_report, spatial, dimming, context = run_design(
                mock_data,
                "Design lighting for a 200m pathway in Fitzroy Gardens with high evening traffic",
            )

        assert design.num_lights > 0
        assert design.spacing_m > 0
        assert design.annual_energy_cost_aud > 0
        assert "LIGHTING DESIGN CALCULATION REPORT" in calc_report
        assert spatial is not None

    def test_pipeline_with_explicit_params(self):
        """Pipeline with explicit parameters bypasses query parsing."""
        from core.pipeline import run_design

        mock_data = _make_mock_appdata()

        with patch("core.pipeline.resolve_osm_pathway", return_value=None):
            design, calc_report, spatial, dimming, context = run_design(
                mock_data,
                "test query",
                location="Melbourne CBD",
                pathway_length_m=300,
                pathway_width_m=4.0,
                activity_level="high",
                location_type="shared_path",
            )

        assert design.pathway_length_m == 300
        assert design.pathway_width_m == 4.0
        assert design.p_category in ("P1", "P2", "P3")

    def test_pipeline_osm_failure_graceful(self):
        """When OSM fails, pipeline uses fallback length."""
        from core.pipeline import run_design

        mock_data = _make_mock_appdata()

        with patch("core.pipeline.resolve_osm_pathway", return_value=None):
            design, calc_report, spatial, dimming, context = run_design(
                mock_data,
                "Design lighting for a 200m pathway in Fitzroy Gardens",
            )

        assert design.num_lights > 0
        assert "Not available" in context


# ============================================================
# E2E: Safety context integration
# ============================================================

class TestE2ESafety:
    def test_safety_context_included(self):
        """Safety context should be attached to spatial dict."""
        from core.pipeline import run_design

        mock_data = _make_mock_appdata()

        with patch("core.pipeline.resolve_osm_pathway", return_value=None):
            design, calc_report, spatial, dimming, context = run_design(
                mock_data,
                "Design lighting for Carlton Gardens",
            )

        assert "safety_context" in spatial
        safety = spatial["safety_context"]
        assert safety is not None
        assert "risk_category" in safety


# ============================================================
# E2E: Intent classification
# ============================================================

class TestE2EIntent:
    def test_new_design_classified(self):
        from core.intent import classify_intent

        result = classify_intent(
            "Design lighting for a 200m pathway in Royal Park",
            has_existing_design=False,
        )
        assert result["intent"] == "new_design"

    def test_modification_classified(self):
        from core.intent import classify_intent

        result = classify_intent(
            "make it brighter and add more lights",
            has_existing_design=True,
        )
        assert result["intent"] == "modify"

    def test_question_classified(self):
        from core.intent import classify_intent

        result = classify_intent(
            "what standard requires this spacing?",
            has_existing_design=True,
        )
        assert result["intent"] == "question"


# ============================================================
# E2E: Validator on real design output
# ============================================================

class TestE2EValidator:
    def test_validate_real_report(self):
        """Validate a report that matches the design data exactly."""
        from llm.calculation_engine import design_lighting, format_design_report
        from core.validator import validate_report

        design = design_lighting("Fitzroy Gardens", 200, 3.0, "high", "park_path")
        report = format_design_report(design)

        result = validate_report(report, design.summary_dict())
        # The format_design_report should match the engine's numbers exactly
        assert result["needs_regeneration"] is False


# ============================================================
# E2E: Budget constraint
# ============================================================

class TestE2EBudget:
    def test_budget_constraint_in_pipeline(self):
        """Budget constraint should produce an alternative when exceeded."""
        from llm.calculation_engine import design_lighting

        design = design_lighting(
            "Test Park", 200, 3.0, "high", "park_path",
            budget_cap=10.0,  # Very tight budget
        )
        assert design.budget_analysis["within_budget"] is False
        assert design.budget_analysis["budget_alternative"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
