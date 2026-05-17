"""
Tests for multi-turn conversation intent classification.
Tests both the LLM-powered classifier and the regex fallback.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.intent import (
    classify_intent,
    classify_intent_regex,
    classify_intent_llm,
    apply_modification,
)


# ============================================================
# Regex fallback (always available, no LLM needed)
# ============================================================

class TestClassifyIntentRegex:
    def test_new_design_explicit(self):
        result = classify_intent_regex("Design lighting for Royal Park")
        assert result["intent"] == "new_design"

    def test_modify_brighter(self):
        result = classify_intent_regex("make it brighter")
        assert result["intent"] == "modify"
        assert result["modify_target"] == "brightness"

    def test_modify_reduce_cost(self):
        result = classify_intent_regex("reduce cost by 20%")
        assert result["intent"] == "modify"
        assert result["modify_target"] == "cost"

    def test_modify_use_solar(self):
        result = classify_intent_regex("use solar panels instead")
        assert result["intent"] == "modify"
        assert result["modify_target"] == "technology"

    def test_modify_add_lights(self):
        result = classify_intent_regex("add more lights near the entrance")
        assert result["intent"] == "modify"
        assert result["modify_target"] == "light_count"

    def test_question_standard(self):
        result = classify_intent_regex("what standard requires this spacing?")
        assert result["intent"] == "question"

    def test_question_explain(self):
        result = classify_intent_regex("explain the P-category selection")
        assert result["intent"] == "question"

    def test_question_why(self):
        result = classify_intent_regex("why did you choose 60W LED?")
        assert result["intent"] == "question"

    def test_ambiguous_defaults_to_question(self):
        result = classify_intent_regex("thanks")
        assert result["intent"] == "question"
        assert result["confidence"] < 0.5


# ============================================================
# LLM-powered classification (mocked)
# ============================================================

class TestClassifyIntentLLM:
    def _mock_llm_response(self, intent, modify_target=None, confidence=0.90):
        """Build a mock LM Studio response."""
        import json as _json
        payload = {"intent": intent, "modify_target": modify_target, "confidence": confidence}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{
                "message": {"content": _json.dumps(payload)}
            }]
        }
        return mock_resp

    def test_llm_classifies_new_design(self):
        with patch("core.intent.requests.post", return_value=self._mock_llm_response("new_design")):
            result = classify_intent_llm("Design lighting for a new park on the south side")
        assert result["intent"] == "new_design"
        assert result["confidence"] == 0.90

    def test_llm_classifies_modify(self):
        with patch("core.intent.requests.post", return_value=self._mock_llm_response("modify", "brightness")):
            result = classify_intent_llm("can we get more light near the benches")
        assert result["intent"] == "modify"
        assert result["modify_target"] == "brightness"

    def test_llm_classifies_question(self):
        with patch("core.intent.requests.post", return_value=self._mock_llm_response("question")):
            result = classify_intent_llm("is this compliant with the latest standard?")
        assert result["intent"] == "question"

    def test_llm_sanitises_invalid_intent(self):
        """If LLM returns an unexpected intent, default to question."""
        with patch("core.intent.requests.post", return_value=self._mock_llm_response("unknown_intent")):
            result = classify_intent_llm("something weird")
        assert result["intent"] == "question"


# ============================================================
# Public API: classify_intent (LLM-first, regex fallback)
# ============================================================

class TestClassifyIntent:
    def test_no_existing_design_always_new(self):
        """Without existing design, skip LLM entirely — always new_design."""
        result = classify_intent("hello world", has_existing_design=False)
        assert result["intent"] == "new_design"
        assert result["confidence"] == 0.95

    def test_falls_back_to_regex_when_llm_unavailable(self):
        """When LLM fails, regex fallback should still classify correctly."""
        with patch("core.intent.classify_intent_llm", side_effect=ConnectionError("LM Studio down")):
            result = classify_intent("make it brighter", has_existing_design=True)
        assert result["intent"] == "modify"
        assert result["modify_target"] == "brightness"

    def test_uses_llm_when_available(self):
        """When LLM is available, its result should be used."""
        llm_result = {"intent": "modify", "modify_target": "cost", "confidence": 0.92}
        with patch("core.intent.classify_intent_llm", return_value=llm_result):
            result = classify_intent("I think the annual cost is too high", has_existing_design=True)
        assert result["intent"] == "modify"
        assert result["modify_target"] == "cost"
        assert result["confidence"] == 0.92


# ============================================================
# Modification application
# ============================================================

SAMPLE_DESIGN = {
    "pathway_length_m": 200,
    "spacing_m": 17.5,
    "annual_energy_cost_aud": 604.80,
    "p_category": "P3",
}


class TestApplyModification:
    def test_brighter_increases_activity(self):
        mod = {"modify_target": "brightness", "message": "make it brighter"}
        overrides = apply_modification(SAMPLE_DESIGN, mod)
        assert overrides.get("activity_level") == "high"

    def test_dimmer_decreases_activity(self):
        mod = {"modify_target": "brightness", "message": "make it dimmer"}
        overrides = apply_modification(SAMPLE_DESIGN, mod)
        assert overrides.get("activity_level") == "low"

    def test_reduce_cost_sets_budget(self):
        mod = {"modify_target": "cost", "message": "reduce cost by 20%"}
        overrides = apply_modification(SAMPLE_DESIGN, mod)
        assert "budget_cap" in overrides
        assert overrides["budget_cap"] == pytest.approx(604.80 * 0.8, rel=0.01)

    def test_wider_spacing(self):
        mod = {"modify_target": "spacing", "message": "use wider spacing"}
        overrides = apply_modification(SAMPLE_DESIGN, mod)
        assert overrides.get("activity_level") == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
