"""
Validates LLM-generated reports against the deterministic calculation engine output.
Ensures numerical claims in the narrative match the authoritative design values.
"""

import re
from typing import Optional


# Patterns for extracting numbers from LLM markdown reports
_PATTERNS = {
    "light_count": [
        r"(\d+)\s*(?:lights?|luminaires?|poles?|fittings?)",
        r"(?:number of lights|light count|total lights)[:\s]*(\d+)",
    ],
    "spacing_m": [
        r"(\d+(?:\.\d+)?)\s*m(?:etre)?s?\s*(?:spacing|apart|intervals?)",
        r"(?:spacing|interval)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*m",
        r"(?:spacing|interval)[:\s]*(\d+(?:\.\d+)?)\s*m",
    ],
    "annual_energy_cost": [
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:/\s*year|per\s*year|annually|p\.a\.|\/yr)",
        r"(?:annual\s*(?:energy\s*)?cost)[:\s]*\$\s*([\d,]+(?:\.\d+)?)",
    ],
    "annual_energy_kwh": [
        r"([\d,]+(?:\.\d+)?)\s*kWh(?:\s*(?:/\s*year|per\s*year|annually|p\.a\.))?",
    ],
    "p_category": [
        r"(?:category\s*)?P(\d+)",
        r"P-?(\d+)",
    ],
}


def extract_numbers_from_report(report_markdown: str) -> dict:
    """
    Extract key numerical claims from an LLM-generated report.

    Returns dict with keys: light_count, spacing_m, annual_energy_cost,
    annual_energy_kwh, p_category. Values are float/int or None if not found.
    """
    results = {}

    for field, patterns in _PATTERNS.items():
        value = None
        for pattern in patterns:
            match = re.search(pattern, report_markdown, re.IGNORECASE)
            if match:
                raw = match.group(1).replace(",", "")
                try:
                    if field in ("light_count", "p_category"):
                        value = int(raw)
                    else:
                        value = float(raw)
                except ValueError:
                    continue
                break
        results[field] = value

    return results


def _get_engine_values(design_data: dict) -> dict:
    """Extract comparable values from the calculation engine's design dict."""
    return {
        "light_count": design_data.get("num_lights"),
        "spacing_m": design_data.get("spacing_m"),
        "annual_energy_cost": design_data.get("annual_energy_cost_aud"),
        "annual_energy_kwh": design_data.get("annual_energy_kwh"),
        "p_category": int(design_data.get("p_category", "P0").replace("P", "")) if design_data.get("p_category") else None,
    }


def validate_report(
    report: str,
    design_data: dict,
    tolerance: float = 0.05,
) -> dict:
    """
    Compare extracted numbers from the LLM report against authoritative design data.

    Args:
        report: LLM-generated markdown report.
        design_data: Dict from LightingDesign.summary_dict() or equivalent.
        tolerance: Fraction tolerance for numeric comparison (0.05 = 5%).

    Returns:
        Dict with: valid, mismatches, corrected_report, needs_regeneration.
    """
    extracted = extract_numbers_from_report(report)
    engine = _get_engine_values(design_data)

    mismatches = []
    corrected_report = report

    for field, report_val in extracted.items():
        if report_val is None:
            continue  # Not mentioned in report, skip

        engine_val = engine.get(field)
        if engine_val is None:
            continue  # No engine value to compare

        # Compare
        if field in ("light_count", "p_category"):
            # Integer comparison — exact match required
            if report_val != engine_val:
                severity = "minor" if abs(report_val - engine_val) <= 1 else "major"
                mismatches.append({
                    "field": field,
                    "report_value": report_val,
                    "engine_value": engine_val,
                    "severity": severity,
                })
        else:
            # Float comparison with tolerance
            if engine_val == 0:
                if report_val != 0:
                    mismatches.append({
                        "field": field,
                        "report_value": report_val,
                        "engine_value": engine_val,
                        "severity": "major",
                    })
                continue

            diff = abs(report_val - engine_val) / abs(engine_val)
            if diff > tolerance * 2:
                mismatches.append({
                    "field": field,
                    "report_value": report_val,
                    "engine_value": engine_val,
                    "severity": "major",
                })
            elif diff > tolerance:
                mismatches.append({
                    "field": field,
                    "report_value": report_val,
                    "engine_value": engine_val,
                    "severity": "minor",
                })

    # Auto-correct minor mismatches in report text
    for m in mismatches:
        if m["severity"] == "minor":
            corrected_report = _apply_correction(
                corrected_report, m["field"], m["report_value"], m["engine_value"]
            )

    has_major = any(m["severity"] == "major" for m in mismatches)

    return {
        "valid": len(mismatches) == 0,
        "mismatches": mismatches,
        "corrected_report": corrected_report if mismatches else None,
        "needs_regeneration": has_major,
    }


def _apply_correction(
    report: str, field: str, old_value, new_value
) -> str:
    """Replace a specific number in the report with the correct value."""
    if field == "light_count":
        report = re.sub(
            rf"\b{old_value}\s*(lights?|luminaires?|poles?)",
            f"{new_value} \\1",
            report,
            count=1,
        )
    elif field == "spacing_m":
        report = re.sub(
            rf"\b{old_value}\s*m",
            f"{new_value}m",
            report,
            count=1,
        )
    elif field == "annual_energy_cost":
        old_str = f"${old_value:.2f}" if isinstance(old_value, float) else f"${old_value}"
        new_str = f"${new_value:.2f}" if isinstance(new_value, float) else f"${new_value}"
        report = report.replace(old_str, new_str, 1)
    elif field == "p_category":
        report = re.sub(
            rf"P{old_value}\b",
            f"P{new_value}",
            report,
            count=1,
        )
    return report


def append_verification_footer(report: str, design_data: dict) -> str:
    """
    Add a Calculation Verification section to the end of the report.
    Shows the authoritative values from the calculation engine.
    """
    footer = (
        "\n\n---\n"
        "## Calculation Verification\n\n"
        "The following values are from the deterministic AS/NZS 1158 calculation engine "
        "and are authoritative:\n\n"
        "| Parameter | Value |\n"
        "|-----------|-------|\n"
        f"| P-Category | {design_data.get('p_category', 'N/A')} |\n"
        f"| Number of Lights | {design_data.get('num_lights', 'N/A')} |\n"
        f"| Spacing | {design_data.get('spacing_m', 'N/A')}m |\n"
        f"| LED Wattage | {design_data.get('led_wattage', 'N/A')}W |\n"
        f"| Annual Energy Cost | ${design_data.get('annual_energy_cost_aud', 'N/A')} |\n"
        f"| Annual CO2 | {design_data.get('annual_co2_kg', 'N/A')} kg |\n"
        f"| Capital Cost | ${design_data.get('total_capital_cost_aud', 'N/A')} |\n"
        f"| Energy Saving vs HPS | {design_data.get('energy_saving_vs_hps_percent', 'N/A')}% |\n"
    )
    return report + footer
