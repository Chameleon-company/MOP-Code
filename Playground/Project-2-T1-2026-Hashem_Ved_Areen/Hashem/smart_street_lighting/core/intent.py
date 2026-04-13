"""
Intent classification for multi-turn design conversations.
Determines whether a message is a new design request, a modification,
or a question about an existing design.

Uses the fast LLM (same parse model as query parsing) for accurate
classification, with regex fallback if LLM is unavailable.
"""

import json
import re
from typing import Optional

import requests

from core.config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_PARSE_MODEL,
    FAST_LLM_TIMEOUT,
    INTENT_MAX_TOKENS,
    FAST_LLM_TEMPERATURE,
)


# ============================================================
# LLM-powered intent classification
# ============================================================

INTENT_SYSTEM_PROMPT = """\
You are a street lighting design assistant. Classify the user's message intent \
given that there is an EXISTING lighting design in the conversation.

Return ONLY a JSON object with these fields:

{
  "intent": "new_design" | "modify" | "question",
  "modify_target": string or null,
  "confidence": number between 0 and 1
}

Intent definitions:
- "new_design": The user wants a completely new design for a DIFFERENT location \
or scenario. Look for: "design lighting for...", "plan lighting for...", \
"new design", or a clearly different park/location name.
- "modify": The user wants to CHANGE the existing design. Look for: \
"make it brighter/dimmer", "reduce cost", "add more lights", "use solar", \
"wider/closer spacing", "cheaper", "increase/decrease", "change to", "upgrade".
- "question": The user is ASKING about the existing design or standards. \
Look for: "what standard", "why did you", "explain", "how does", "what does X mean", \
"tell me about", or any interrogative about the current design.

For "modify" intent, set modify_target to one of:
- "brightness" (brighter, dimmer, darker, more/less light)
- "cost" (cheaper, reduce cost, budget, save money)
- "spacing" (closer, wider, further apart)
- "technology" (solar, LED type change, different luminaire)
- "light_count" (more lights, fewer lights, add/remove)
- "general" (other modification)

For "new_design" and "question", set modify_target to null.

Return ONLY valid JSON. No markdown, no explanation."""


def classify_intent_llm(message: str) -> dict:
    """Use the fast LLM to classify intent. Called only when existing design exists."""
    resp = requests.post(
        f"{LM_STUDIO_BASE_URL}/chat/completions",
        json={
            "model": LM_STUDIO_PARSE_MODEL,
            "messages": [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "max_tokens": INTENT_MAX_TOKENS,
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

    intent = parsed.get("intent", "question")
    if intent not in ("new_design", "modify", "question"):
        intent = "question"

    return {
        "intent": intent,
        "modify_target": parsed.get("modify_target"),
        "confidence": float(parsed.get("confidence", 0.85)),
    }


# ============================================================
# Regex fallback (no LLM needed)
# ============================================================

NEW_DESIGN_KEYWORDS = [
    r"\bdesign\s+(?:lighting|lights?)\s+for\b",
    r"\bplan\s+lighting\b",
    r"\brecommend\s+lighting\b",
    r"\bcreate\s+(?:a\s+)?lighting\b",
    r"\bnew\s+(?:design|lighting)\b",
]

MODIFY_KEYWORDS = [
    r"\b(?:make|go)\s+(?:it\s+)?(?:brighter|dimmer|darker)\b",
    r"\breduce\s+(?:cost|budget|spacing|lights)\b",
    r"\bincrease\s+(?:spacing|brightness|lights)\b",
    r"\badd\s+(?:more\s+)?lights?\b",
    r"\bfewer\s+lights?\b",
    r"\buse\s+solar\b",
    r"\bwider\s+spacing\b",
    r"\bcloser\s+spacing\b",
    r"\bchange\s+(?:to|the)\b",
    r"\bswitch\s+to\b",
    r"\bupgrade\b",
    r"\bdowngrade\b",
    r"\breduce.*budget\b",
    r"\bcut.*cost\b",
    r"\bcheaper\b",
    r"\bmore\s+energy.?efficient\b",
]

QUESTION_KEYWORDS = [
    r"\bwhat\s+(?:standard|category|is|does|are)\b",
    r"\bwhy\s+(?:did|do|is|was|were)\b",
    r"\bexplain\b",
    r"\bhow\s+(?:does|did|is|was|many|much)\b",
    r"\btell\s+me\s+about\b",
    r"\bwhat.+mean\b",
    r"\bcan\s+you\s+explain\b",
]


def classify_intent_regex(message: str) -> dict:
    """Regex-based fallback for intent classification."""
    msg = message.lower().strip()

    for pattern in NEW_DESIGN_KEYWORDS:
        if re.search(pattern, msg):
            return {"intent": "new_design", "modify_target": None, "confidence": 0.75}

    for pattern in MODIFY_KEYWORDS:
        if re.search(pattern, msg):
            target = _extract_modify_target(msg)
            return {"intent": "modify", "modify_target": target, "confidence": 0.70}

    for pattern in QUESTION_KEYWORDS:
        if re.search(pattern, msg):
            return {"intent": "question", "modify_target": None, "confidence": 0.70}

    return {"intent": "question", "modify_target": None, "confidence": 0.40}


# ============================================================
# Public API
# ============================================================

def classify_intent(message: str, has_existing_design: bool) -> dict:
    """
    Classify the intent of a user message in the context of a design conversation.

    Uses LLM for accurate classification (handles nuance, sarcasm, implicit requests).
    Falls back to regex if LLM is unavailable.

    Args:
        message: The user's message text.
        has_existing_design: Whether there's an existing design in the conversation.

    Returns:
        Dict with "intent" (new_design|modify|question), "modify_target" (str|None),
        and "confidence" (float).
    """
    # If no existing design, everything is a new design request
    if not has_existing_design:
        return {"intent": "new_design", "modify_target": None, "confidence": 0.95}

    # Try LLM first, fall back to regex
    try:
        return classify_intent_llm(message)
    except Exception as e:
        print(f"LLM intent classification failed ({e}), using regex fallback.")
        return classify_intent_regex(message)


def _extract_modify_target(msg: str) -> str:
    """Extract what the user wants to modify (used by regex fallback)."""
    if any(w in msg for w in ["brighter", "dimmer", "darker", "brightness"]):
        return "brightness"
    if any(w in msg for w in ["cost", "budget", "cheaper", "expensive"]):
        return "cost"
    if any(w in msg for w in ["spacing", "closer", "wider", "apart"]):
        return "spacing"
    if any(w in msg for w in ["solar", "renewable"]):
        return "technology"
    if any(w in msg for w in ["more lights", "add lights", "fewer lights"]):
        return "light_count"
    return "general"


def apply_modification(existing_design: dict, modification: dict) -> dict:
    """
    Apply a requested change to existing design parameters.

    Args:
        existing_design: The current design summary dict.
        modification: Dict with "modify_target" and original message context.

    Returns:
        Dict of overridden parameters for the calculation engine.
    """
    target = modification.get("modify_target", "general")
    overrides = {}

    if target == "brightness":
        msg = modification.get("message", "").lower()
        if "brighter" in msg:
            overrides["pathway_length_m"] = existing_design.get("pathway_length_m", 200)
            overrides["activity_level"] = "high"
        elif "dimmer" in msg or "darker" in msg:
            overrides["activity_level"] = "low"

    elif target == "cost":
        msg = modification.get("message", "").lower()
        pct_match = re.search(r"(\d+)\s*%", msg)
        if pct_match:
            reduction_pct = int(pct_match.group(1))
            current_cost = existing_design.get("annual_energy_cost_aud", 500)
            overrides["budget_cap"] = current_cost * (1 - reduction_pct / 100)
        else:
            current_cost = existing_design.get("annual_energy_cost_aud", 500)
            overrides["budget_cap"] = current_cost * 0.8

    elif target == "technology":
        # TODO: Support technology switching (e.g., to solar) in a future iteration
        pass

    elif target == "spacing":
        msg = modification.get("message", "").lower()
        if "closer" in msg or "tighter" in msg:
            overrides["activity_level"] = "high"
        elif "wider" in msg:
            overrides["activity_level"] = "low"

    return overrides
