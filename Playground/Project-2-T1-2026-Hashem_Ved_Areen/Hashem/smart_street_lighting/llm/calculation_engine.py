"""
Lighting Calculation Engine.

Deterministic calculations for street lighting design based on
AS/NZS 1158 standards. This is the core computation layer — the LLM
explains and justifies, but the numbers come from here.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# AS/NZS 1158 P-Category Standards Lookup
# ============================================================

P_CATEGORIES = {
    "P1": {
        "name": "Major pedestrian activity",
        "avg_lux": 14.0,
        "min_lux": 7.0,
        "uniformity": 0.50,
    },
    "P2": {
        "name": "High-activity pedestrian",
        "avg_lux": 10.0,
        "min_lux": 5.0,
        "uniformity": 0.50,
    },
    "P3": {
        "name": "Moderate pedestrian",
        "avg_lux": 7.0,
        "min_lux": 3.5,
        "uniformity": 0.50,
    },
    "P4": {
        "name": "Moderate-low pedestrian",
        "avg_lux": 5.0,
        "min_lux": 2.5,
        "uniformity": 0.50,
    },
    "P5": {
        "name": "Low pedestrian",
        "avg_lux": 3.5,
        "min_lux": 1.75,
        "uniformity": 0.50,
    },
    "P6": {
        "name": "Low-activity pedestrian",
        "avg_lux": 3.5,
        "min_lux": 0.75,
        "uniformity": 0.21,
    },
    "P7": {
        "name": "Minor pedestrian",
        "avg_lux": 1.5,
        "min_lux": 0.75,
        "uniformity": 0.50,
    },
    "P8": {
        "name": "Minor pedestrian (low risk)",
        "avg_lux": 1.5,
        "min_lux": 0.38,
        "uniformity": 0.25,
    },
    "P9": {
        "name": "Park paths (moderate use)",
        "avg_lux": 2.0,
        "min_lux": 1.0,
        "uniformity": 0.50,
    },
    "P10": {
        "name": "Park paths (low use)",
        "avg_lux": 1.0,
        "min_lux": 0.50,
        "uniformity": 0.50,
    },
    "P11": {
        "name": "Outdoor car parks (commercial)",
        "avg_lux": 7.0,
        "min_lux": 1.75,
        "uniformity": 0.25,
    },
    "P12": {
        "name": "Outdoor car parks (residential)",
        "avg_lux": 3.5,
        "min_lux": 0.88,
        "uniformity": 0.25,
    },
}

# ============================================================
# LED Technology Specs (typical values for Melbourne)
# ============================================================

LED_SPECS = {
    "low": {
        "wattage": 30,
        "lumens": 4500,
        "description": "30W LED (park path bollard/low-mount)",
    },
    "medium": {
        "wattage": 60,
        "lumens": 9000,
        "description": "60W LED (pedestrian area standard)",
    },
    "high": {
        "wattage": 100,
        "lumens": 15000,
        "description": "100W LED (major pathway/road)",
    },
    "very_high": {
        "wattage": 150,
        "lumens": 22500,
        "description": "150W LED (intersection/high-activity)",
    },
}

# ============================================================
# Melbourne Energy Constants
# ============================================================

OPERATING_HOURS_PER_YEAR = 4200  # dusk to dawn average for Melbourne
ELECTRICITY_RATE_PER_KWH = 0.20  # AUD, mid-range Victorian rate
CARBON_FACTOR_VIC_SCOPE2_3 = 1.08  # kg CO2-e per kWh (Scope 2: 0.96 + Scope 3: 0.12)
RECOMMENDED_CCT = 3000  # Kelvin (warm white, Melbourne ecological guideline)
LED_MAINTENANCE_FACTOR = 0.87  # typical for LED luminaires
LED_LIFESPAN_YEARS = 20
LED_CRI = 70


@dataclass
class LightingDesign:
    """Complete lighting design output from the calculation engine."""

    # Input parameters
    location_name: str
    pathway_length_m: float
    pathway_width_m: float = 3.0
    activity_level: str = "moderate"  # low, moderate, high, very_high

    # Category selection
    p_category: str = ""
    category_name: str = ""
    required_avg_lux: float = 0.0
    required_min_lux: float = 0.0
    required_uniformity: float = 0.0

    # Design specs
    pole_height_m: float = 0.0
    spacing_m: float = 0.0
    num_lights: int = 0
    led_spec: str = ""
    led_wattage: int = 0
    led_lumens: int = 0
    colour_temperature_k: int = RECOMMENDED_CCT

    # Energy & cost estimates
    total_system_wattage: float = 0.0
    annual_energy_kwh: float = 0.0
    annual_energy_cost_aud: float = 0.0
    annual_co2_kg: float = 0.0
    capital_cost_per_light_aud: float = 0.0
    total_capital_cost_aud: float = 0.0
    annual_maintenance_cost_aud: float = 0.0

    # Comparison vs HPS baseline
    hps_equivalent_wattage: int = 0
    hps_annual_energy_kwh: float = 0.0
    hps_annual_cost_aud: float = 0.0
    energy_saving_percent: float = 0.0
    co2_saving_kg: float = 0.0
    payback_years: float = 0.0

    # Enhanced: geometry-aware placement, budget, safety
    light_positions: list = field(default_factory=list)
    budget_analysis: dict = field(default_factory=dict)
    safety_adjustment_applied: int = 0
    pathway_geometry: dict = field(default_factory=dict)

    def summary_dict(self) -> dict:
        """Return key outputs as a dictionary for LLM context."""
        d = {
            "location": self.location_name,
            "pathway_length_m": self.pathway_length_m,
            "p_category": self.p_category,
            "category_name": self.category_name,
            "required_avg_lux": self.required_avg_lux,
            "num_lights": self.num_lights,
            "spacing_m": self.spacing_m,
            "pole_height_m": self.pole_height_m,
            "led_wattage": self.led_wattage,
            "colour_temperature": f"{self.colour_temperature_k}K",
            "annual_energy_cost_aud": round(self.annual_energy_cost_aud, 2),
            "annual_energy_kwh": round(self.annual_energy_kwh, 1),
            "annual_co2_kg": round(self.annual_co2_kg, 1),
            "total_capital_cost_aud": round(self.total_capital_cost_aud, 2),
            "energy_saving_vs_hps_percent": round(self.energy_saving_percent, 1),
            "co2_saving_vs_hps_kg": round(self.co2_saving_kg, 1),
            "payback_years": round(self.payback_years, 1),
        }
        if self.light_positions:
            d["light_positions"] = self.light_positions
        if self.budget_analysis:
            d["budget_analysis"] = self.budget_analysis
        if self.safety_adjustment_applied:
            d["safety_adjustment_applied"] = self.safety_adjustment_applied
        return d


def select_p_category(activity_level: str, location_type: str = "park_path") -> str:
    """
    Select the appropriate P-category based on activity level and location type.

    Args:
        activity_level: "low", "moderate", "high", "very_high"
        location_type: "park_path", "shared_path", "public_space", "residential"

    Returns:
        P-category string (e.g., "P3")
    """
    if location_type == "park_path":
        mapping = {"low": "P10", "moderate": "P9", "high": "P3", "very_high": "P2"}
    elif location_type == "shared_path":
        mapping = {"low": "P5", "moderate": "P3", "high": "P2", "very_high": "P1"}
    elif location_type == "public_space":
        mapping = {"low": "P5", "moderate": "P3", "high": "P2", "very_high": "P1"}
    elif location_type == "residential":
        mapping = {"low": "P8", "moderate": "P6", "high": "P5", "very_high": "P4"}
    else:
        mapping = {"low": "P10", "moderate": "P9", "high": "P3", "very_high": "P2"}

    return mapping.get(activity_level, "P9")


def select_led_spec(p_category: str) -> str:
    """Select appropriate LED spec based on the lighting category."""
    high_cats = {"P1", "P2", "P11"}
    medium_cats = {"P3", "P4", "P5"}
    if p_category in high_cats:
        return "high"
    elif p_category in medium_cats:
        return "medium"
    else:
        return "low"


def calculate_spacing(pole_height: float, p_category: str) -> float:
    """
    Calculate recommended spacing based on pole height and category.
    Rule: spacing = multiplier × pole_height (AS/NZS 1158 guidance: 3-5x).
    """
    # Higher categories need closer spacing for uniformity
    multipliers = {
        "P1": 3.0,
        "P2": 3.5,
        "P3": 3.5,
        "P4": 4.0,
        "P5": 4.0,
        "P6": 4.5,
        "P7": 5.0,
        "P8": 5.0,
        "P9": 4.0,
        "P10": 5.0,
        "P11": 4.0,
        "P12": 4.5,
    }
    mult = multipliers.get(p_category, 4.0)
    return round(pole_height * mult, 1)


def select_pole_height(p_category: str, pathway_width: float) -> float:
    """Select pole height based on category and pathway width."""
    if p_category in {"P1", "P2"}:
        return 6.0 if pathway_width >= 3.0 else 5.0
    elif p_category in {"P3", "P4", "P5"}:
        return 5.0 if pathway_width >= 3.0 else 4.0
    else:
        return 4.0 if pathway_width >= 2.0 else 3.5


def design_lighting(
    location_name: str,
    pathway_length_m: float,
    pathway_width_m: float = 3.0,
    activity_level: str = "moderate",
    location_type: str = "park_path",
    avg_pedestrian_count: Optional[float] = None,
    safety_adjustment: int = 0,
    pathway_geometry: Optional[dict] = None,
    intersections: Optional[list] = None,
    entry_points: Optional[list] = None,
    budget_cap: Optional[float] = None,
) -> LightingDesign:
    """
    Complete lighting design calculation for a pathway.

    This is the main entry point for the calculation engine.
    Given physical parameters and activity level, it produces a full
    design with light count, spacing, energy cost, and HPS comparison.

    Args:
        location_name: Name of the location.
        pathway_length_m: Length in metres.
        pathway_width_m: Width in metres.
        activity_level: "low", "moderate", "high", "very_high"
        location_type: "park_path", "shared_path", "public_space", "residential"
        avg_pedestrian_count: Average hourly pedestrian count (if known from data).
        safety_adjustment: P-category adjustment from safety analysis (negative = upgrade).
        pathway_geometry: GeoJSON LineString dict for geometry-aware placement.
        intersections: List of intersection dicts from OSM analysis.
        entry_points: List of entry point dicts from OSM analysis.
        budget_cap: Optional annual budget cap in AUD.

    Returns:
        LightingDesign with all calculations populated.
    """
    # If we have real pedestrian data, override activity level
    if avg_pedestrian_count is not None:
        if avg_pedestrian_count < 50:
            activity_level = "low"
        elif avg_pedestrian_count < 300:
            activity_level = "moderate"
        elif avg_pedestrian_count < 1000:
            activity_level = "high"
        else:
            activity_level = "very_high"

    design = LightingDesign(
        location_name=location_name,
        pathway_length_m=pathway_length_m,
        pathway_width_m=pathway_width_m,
        activity_level=activity_level,
    )

    # 1. Select P-category (with optional safety adjustment)
    design.p_category = select_p_category(activity_level, location_type)
    if safety_adjustment != 0:
        p_num = int(design.p_category.replace("P", ""))
        adjusted = max(1, min(12, p_num + safety_adjustment))
        design.p_category = f"P{adjusted}"
    cat = P_CATEGORIES[design.p_category]
    design.category_name = cat["name"]
    design.required_avg_lux = cat["avg_lux"]
    design.required_min_lux = cat["min_lux"]
    design.required_uniformity = cat["uniformity"]

    # 2. Select pole height and spacing
    design.pole_height_m = select_pole_height(design.p_category, pathway_width_m)
    design.spacing_m = calculate_spacing(design.pole_height_m, design.p_category)

    # 3. Calculate number of lights (fence-post: lights at 0, spacing, 2*spacing, ..., end)
    design.num_lights = max(2, math.floor(pathway_length_m / design.spacing_m) + 1)

    # 4. Select LED technology
    design.led_spec = select_led_spec(design.p_category)
    spec = LED_SPECS[design.led_spec]
    design.led_wattage = spec["wattage"]
    design.led_lumens = spec["lumens"]

    # 5. Energy calculations (LED)
    design.total_system_wattage = design.num_lights * design.led_wattage
    design.annual_energy_kwh = (
        design.total_system_wattage * OPERATING_HOURS_PER_YEAR
    ) / 1000
    design.annual_energy_cost_aud = design.annual_energy_kwh * ELECTRICITY_RATE_PER_KWH
    design.annual_co2_kg = design.annual_energy_kwh * CARBON_FACTOR_VIC_SCOPE2_3

    # 6. Capital cost estimate
    # Capital cost includes luminaire + pole + installation + wiring
    # (luminaire-only costs are 30-40% of total installed cost)
    cost_per_light_installed = {
        "low": 3000,
        "medium": 4500,
        "high": 6000,
        "very_high": 8000,
    }
    design.capital_cost_per_light_aud = cost_per_light_installed[design.led_spec]
    design.total_capital_cost_aud = (
        design.num_lights * design.capital_cost_per_light_aud
    )
    design.annual_maintenance_cost_aud = (
        design.num_lights * 15
    )  # ~$15/light/year for LED

    # 7. HPS baseline comparison
    hps_wattage_map = {"low": 70, "medium": 175, "high": 250, "very_high": 400}
    design.hps_equivalent_wattage = hps_wattage_map[design.led_spec]
    hps_total_w = design.num_lights * design.hps_equivalent_wattage
    design.hps_annual_energy_kwh = (hps_total_w * OPERATING_HOURS_PER_YEAR) / 1000
    design.hps_annual_cost_aud = design.hps_annual_energy_kwh * ELECTRICITY_RATE_PER_KWH

    # 8. Savings
    if design.hps_annual_energy_kwh > 0:
        design.energy_saving_percent = (
            (design.hps_annual_energy_kwh - design.annual_energy_kwh)
            / design.hps_annual_energy_kwh
            * 100
        )
    design.co2_saving_kg = (
        design.hps_annual_energy_kwh - design.annual_energy_kwh
    ) * CARBON_FACTOR_VIC_SCOPE2_3

    # 9. Payback period (retrofit scenario — luminaire replacement only)
    # For retrofits, cost is luminaire + installation only (poles already exist)
    retrofit_cost_per_light = {
        "low": 1000,
        "medium": 1500,
        "high": 2000,
        "very_high": 2800,
    }
    retrofit_total = design.num_lights * retrofit_cost_per_light[design.led_spec]
    annual_saving = (design.hps_annual_cost_aud - design.annual_energy_cost_aud) + (
        design.num_lights * 60
    )  # +$60/light/yr HPS maintenance saving (industry avg)
    if annual_saving > 0:
        design.payback_years = retrofit_total / annual_saving

    # 10. Record safety adjustment
    design.safety_adjustment_applied = safety_adjustment

    # 11. Geometry-aware light placement (if pathway geometry provided)
    if pathway_geometry and pathway_geometry.get("coordinates"):
        try:
            from data.geometry import place_lights_on_polyline

            coords_lonlat = pathway_geometry["coordinates"]
            coords = [(c[1], c[0]) for c in coords_lonlat]  # GeoJSON is [lon, lat]
            design.light_positions = place_lights_on_polyline(
                coords,
                design.spacing_m,
                intersections=intersections,
                entry_points=entry_points,
            )
            design.pathway_geometry = pathway_geometry
            # Update light count to match actual placed lights
            if design.light_positions:
                design.num_lights = len(design.light_positions)
                # Recalculate energy/cost with updated light count
                design.total_system_wattage = design.num_lights * design.led_wattage
                design.annual_energy_kwh = (
                    design.total_system_wattage * OPERATING_HOURS_PER_YEAR
                ) / 1000
                design.annual_energy_cost_aud = (
                    design.annual_energy_kwh * ELECTRICITY_RATE_PER_KWH
                )
                design.annual_co2_kg = (
                    design.annual_energy_kwh * CARBON_FACTOR_VIC_SCOPE2_3
                )
                design.total_capital_cost_aud = (
                    design.num_lights * design.capital_cost_per_light_aud
                )
                design.annual_maintenance_cost_aud = design.num_lights * 15
                hps_total_w = design.num_lights * design.hps_equivalent_wattage
                design.hps_annual_energy_kwh = (
                    hps_total_w * OPERATING_HOURS_PER_YEAR
                ) / 1000
                design.hps_annual_cost_aud = (
                    design.hps_annual_energy_kwh * ELECTRICITY_RATE_PER_KWH
                )
                if design.hps_annual_energy_kwh > 0:
                    design.energy_saving_percent = (
                        (design.hps_annual_energy_kwh - design.annual_energy_kwh)
                        / design.hps_annual_energy_kwh
                        * 100
                    )
                design.co2_saving_kg = (
                    design.hps_annual_energy_kwh - design.annual_energy_kwh
                ) * CARBON_FACTOR_VIC_SCOPE2_3
                retrofit_total = (
                    design.num_lights * retrofit_cost_per_light[design.led_spec]
                )
                annual_saving = (
                    design.hps_annual_cost_aud - design.annual_energy_cost_aud
                ) + (design.num_lights * 60)
                if annual_saving > 0:
                    design.payback_years = retrofit_total / annual_saving
        except Exception as e:
            print(f"Geometry placement failed, using linear calculation: {e}")

    # 12. Budget analysis
    if budget_cap is not None:
        total_annual = (
            design.annual_energy_cost_aud + design.annual_maintenance_cost_aud
        )
        within_budget = total_annual <= budget_cap

        budget_alt = None
        compliance_notes = []
        if not within_budget:
            # Try wider spacing (up to 1.3x) and lower wattage
            alt_spacing = design.spacing_m * 1.3
            alt_num_lights = max(2, math.floor(pathway_length_m / alt_spacing) + 1)
            # Try one LED spec level lower
            spec_order = ["very_high", "high", "medium", "low"]
            current_idx = (
                spec_order.index(design.led_spec)
                if design.led_spec in spec_order
                else -1
            )
            alt_spec_key = (
                spec_order[min(current_idx + 1, len(spec_order) - 1)]
                if current_idx >= 0
                else design.led_spec
            )
            alt_spec = LED_SPECS[alt_spec_key]

            alt_energy = (
                alt_num_lights * alt_spec["wattage"] * OPERATING_HOURS_PER_YEAR
            ) / 1000
            alt_cost = alt_energy * ELECTRICITY_RATE_PER_KWH
            alt_maint = alt_num_lights * 15
            alt_total = alt_cost + alt_maint

            budget_alt = {
                "num_lights": alt_num_lights,
                "spacing_m": round(alt_spacing, 1),
                "led_wattage": alt_spec["wattage"],
                "annual_energy_cost_aud": round(alt_cost, 2),
                "annual_total_cost_aud": round(alt_total, 2),
            }
            compliance_notes.append(
                f"Budget alternative uses {alt_spacing:.1f}m spacing (1.3x standard), "
                f"which may reduce uniformity below AS/NZS 1158 requirements."
            )

        design.budget_analysis = {
            "budget_cap": budget_cap,
            "within_budget": within_budget,
            "full_design_annual_cost": round(total_annual, 2),
            "budget_alternative": budget_alt,
            "compliance_notes": compliance_notes,
        }

    return design


def format_design_report(design: LightingDesign) -> str:
    """Format a human-readable design report from calculations."""
    return f"""
LIGHTING DESIGN CALCULATION REPORT
{'='*50}
Location: {design.location_name}
Pathway: {design.pathway_length_m}m long x {design.pathway_width_m}m wide
Activity Level: {design.activity_level}

CATEGORY SELECTION
  AS/NZS 1158 Category: {design.p_category} — {design.category_name}
  Required average illuminance: {design.required_avg_lux} lux
  Required minimum illuminance: {design.required_min_lux} lux
  Required uniformity (Emin/Eavg): {design.required_uniformity}

DESIGN SPECIFICATIONS
  Number of lights: {design.num_lights}
  Spacing: {design.spacing_m}m
  Pole height: {design.pole_height_m}m
  Technology: {design.led_wattage}W LED ({design.led_lumens} lumens)
  Colour temperature: {design.colour_temperature_k}K (warm white)
  CRI: {LED_CRI}

ENERGY & COST ESTIMATES (LED)
  Total system wattage: {design.total_system_wattage}W
  Annual energy: {design.annual_energy_kwh:.0f} kWh
  Annual energy cost: ${design.annual_energy_cost_aud:.2f}
  Annual CO2 emissions: {design.annual_co2_kg:.1f} kg CO2-e
  Capital cost (new install): ${design.total_capital_cost_aud:,.0f} ({design.num_lights} x ${design.capital_cost_per_light_aud}, includes pole + wiring)

COMPARISON vs HPS BASELINE
  HPS equivalent: {design.hps_equivalent_wattage}W per light
  HPS annual energy cost: ${design.hps_annual_cost_aud:.2f}
  Energy saving: {design.energy_saving_percent:.1f}%
  CO2 saving: {design.co2_saving_kg:.1f} kg CO2-e/year
  Retrofit payback period: {design.payback_years:.1f} years (luminaire swap only)
""".strip()


if __name__ == "__main__":
    # UC-01 test: Fitzroy Gardens pathway
    design = design_lighting(
        location_name="Fitzroy Gardens Main Pathway",
        pathway_length_m=200,
        pathway_width_m=3.0,
        activity_level="high",
        location_type="park_path",
    )
    print(format_design_report(design))
    print()
    print("Summary dict for LLM:")
    for k, v in design.summary_dict().items():
        print(f"  {k}: {v}")
