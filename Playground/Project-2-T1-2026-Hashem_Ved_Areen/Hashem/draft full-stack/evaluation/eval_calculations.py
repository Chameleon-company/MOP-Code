"""
Evaluate the calculation engine against expected design outputs.

Tests that the deterministic calculation engine produces correct results
across different scenarios, and validates edge cases.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.calculation_engine import (
    design_lighting, P_CATEGORIES,
    OPERATING_HOURS_PER_YEAR, ELECTRICITY_RATE_PER_KWH, CARBON_FACTOR_VIC_SCOPE2_3,
)


# Expected design outputs for verification
CALCULATION_TESTS = [
    {
        "id": "CALC-V01",
        "description": "200m moderate park path (UC-01 baseline)",
        "inputs": {"location_name": "Test", "pathway_length_m": 200, "pathway_width_m": 3.0,
                   "activity_level": "moderate", "location_type": "park_path"},
        "expected": {
            "p_category": "P9",
            "num_lights": 13,
            "spacing_m": 16.0,
            "led_wattage": 30,
            "energy_saving_percent_min": 50,
        },
    },
    {
        "id": "CALC-V02",
        "description": "200m high-traffic park path",
        "inputs": {"location_name": "Test", "pathway_length_m": 200, "pathway_width_m": 3.0,
                   "activity_level": "high", "location_type": "park_path"},
        "expected": {
            "p_category": "P3",
            "num_lights": 12,
            "spacing_m": 17.5,
            "led_wattage": 60,
        },
    },
    {
        "id": "CALC-V03",
        "description": "100m low-use park path",
        "inputs": {"location_name": "Test", "pathway_length_m": 100, "pathway_width_m": 2.0,
                   "activity_level": "low", "location_type": "park_path"},
        "expected": {
            "p_category": "P10",
            "led_wattage": 30,
        },
    },
    {
        "id": "CALC-V04",
        "description": "500m very high traffic shared path",
        "inputs": {"location_name": "Test", "pathway_length_m": 500, "pathway_width_m": 4.0,
                   "activity_level": "very_high", "location_type": "shared_path"},
        "expected": {
            "p_category": "P1",
            "led_wattage": 100,
        },
    },
    {
        "id": "CALC-V05",
        "description": "Activity level override from pedestrian count",
        "inputs": {"location_name": "Test", "pathway_length_m": 200, "pathway_width_m": 3.0,
                   "avg_pedestrian_count": 600},
        "expected": {
            "activity_level": "high",
            "p_category": "P3",
        },
    },
    {
        "id": "CALC-V06",
        "description": "Minimum 2 lights for very short path",
        "inputs": {"location_name": "Test", "pathway_length_m": 5, "pathway_width_m": 2.0,
                   "activity_level": "low", "location_type": "park_path"},
        "expected": {
            "num_lights_min": 2,
        },
    },
    {
        "id": "CALC-V07",
        "description": "Energy calculation arithmetic",
        "inputs": {"location_name": "Test", "pathway_length_m": 200, "pathway_width_m": 3.0,
                   "activity_level": "moderate", "location_type": "park_path"},
        "expected": {
            "verify_energy_formula": True,
        },
    },
]


def run_calculation_tests():
    """Run all calculation verification tests."""
    results = []
    passed = 0
    failed = 0

    for test in CALCULATION_TESTS:
        design = design_lighting(**test["inputs"])
        expected = test["expected"]
        errors = []

        # Check each expected field
        if "p_category" in expected:
            if design.p_category != expected["p_category"]:
                errors.append(f"p_category: got {design.p_category}, expected {expected['p_category']}")

        if "num_lights" in expected:
            if design.num_lights != expected["num_lights"]:
                errors.append(f"num_lights: got {design.num_lights}, expected {expected['num_lights']}")

        if "num_lights_min" in expected:
            if design.num_lights < expected["num_lights_min"]:
                errors.append(f"num_lights: got {design.num_lights}, expected >= {expected['num_lights_min']}")

        if "spacing_m" in expected:
            if abs(design.spacing_m - expected["spacing_m"]) > 0.1:
                errors.append(f"spacing_m: got {design.spacing_m}, expected {expected['spacing_m']}")

        if "led_wattage" in expected:
            if design.led_wattage != expected["led_wattage"]:
                errors.append(f"led_wattage: got {design.led_wattage}, expected {expected['led_wattage']}")

        if "activity_level" in expected:
            if design.activity_level != expected["activity_level"]:
                errors.append(f"activity_level: got {design.activity_level}, expected {expected['activity_level']}")

        if "energy_saving_percent_min" in expected:
            if design.energy_saving_percent < expected["energy_saving_percent_min"]:
                errors.append(f"energy_saving: got {design.energy_saving_percent:.1f}%, expected >= {expected['energy_saving_percent_min']}%")

        if expected.get("verify_energy_formula"):
            expected_kwh = (design.num_lights * design.led_wattage * OPERATING_HOURS_PER_YEAR) / 1000
            if abs(design.annual_energy_kwh - expected_kwh) > 0.01:
                errors.append(f"energy formula: got {design.annual_energy_kwh}, expected {expected_kwh}")

            expected_cost = expected_kwh * ELECTRICITY_RATE_PER_KWH
            if abs(design.annual_energy_cost_aud - expected_cost) > 0.01:
                errors.append(f"cost formula: got {design.annual_energy_cost_aud}, expected {expected_cost}")

            expected_co2 = expected_kwh * CARBON_FACTOR_VIC_SCOPE2_3
            if abs(design.annual_co2_kg - expected_co2) > 0.1:
                errors.append(f"co2 formula: got {design.annual_co2_kg}, expected {expected_co2}")

        # Record result
        test_passed = len(errors) == 0
        if test_passed:
            passed += 1
        else:
            failed += 1

        status = "PASS" if test_passed else "FAIL"
        print(f"  [{status}] {test['id']}: {test['description']}")
        for e in errors:
            print(f"         {e}")

        results.append({
            "test_id": test["id"],
            "description": test["description"],
            "passed": test_passed,
            "errors": errors,
            "design_summary": design.summary_dict(),
        })

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(CALCULATION_TESTS)}")
    return results, passed, failed


if __name__ == "__main__":
    print("=" * 60)
    print("CALCULATION ENGINE VERIFICATION")
    print("=" * 60)

    results, passed, failed = run_calculation_tests()

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "eval_calculations.json", "w") as f:
        json.dump({"passed": passed, "failed": failed, "results": results}, f, indent=2)
    print(f"\nResults saved to outputs/eval_calculations.json")
