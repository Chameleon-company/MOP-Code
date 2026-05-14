# UC00219 - Thermal Hotspot vs Tree Canopy Overlay

**Authored by:** Thien Khang Nguyen  
**Duration:** 90 mins  
**Level:** Intermediate  
**Pre-requisite Skills:** Python, Data Cleaning, Data Visualisation, Geospatial Analysis, Raster/Vector Data Handling, Spatial Overlay, Feature Interpretation

## Scenario
Melbourne does not experience urban heat evenly. Some areas show high thermal intensity while others remain relatively cooler, and canopy coverage is one factor that may influence this difference.

This use case addresses:

**Which Melbourne areas combine high relative thermal intensity and very low tree canopy cover, and should therefore be prioritised for greening?**

## Learning Outcomes
At the end of this use case, you can:
- Load and combine public geospatial vector/raster data.
- Align CRS and verify spatial comparability.
- Build a 250 m analysis grid for consistent cell-level indicators.
- Calculate canopy coverage and thermal intensity per cell.
- Apply transparent rule-based hotspot identification.
- Evaluate and validate clustering candidates for canopy-thermal profiles.
- Export handover evidence for Sprint 5 reporting.

## Analysis Questions
1. Where are the highest relative thermal-intensity cells?
2. Which of those cells also have very low canopy cover?
3. Do clustering profiles support the same hotspot patterns found by the rule-based method?

## Method and Justification
- **Grid-based comparison (250 m):** balances spatial detail and stability.
- **Rule-based hotspot logic:** top thermal share + low canopy threshold is transparent and easy to explain.
- **Sensitivity and threshold tuning:** checks robustness against nearby parameter choices.
- **Clustering candidate search:** compares scaler/k/seed configurations for stability and quality.
- **Baseline vs selected comparison:** supports model-selection justification for reporting.
- **Hotspot score v2 + communication labels:** improves prioritisation readability and presentation clarity.

## Data Sources
- **Thermal Image 2012** (City of Melbourne)
- **Tree Canopies 2021 (Urban Forest)** (City of Melbourne)

Both datasets are accessed through public sources/API endpoints without embedding credentials.

## What Was Improved
- Added deeper sensitivity and priority vs non-priority comparative evidence.
- Added model candidate search across scaler/k/seed combinations.
- Added baseline vs selected model comparison summary.
- Added final interpretation and limitations sections for Sprint 5 reporting.
- Added lightweight threshold tuning table and hotspot score v2 ranking aid.
- Added presentation-friendly cluster labels (communication layer).
- Added evidence exports to `outputs/` and a final PASS/FAIL quality checklist.

## Final Interpretation
Hotspot cells represent **relative priority areas** where thermal intensity is high and canopy cover is very low.  
Rule-based hotspots provide a transparent baseline, while clustering adds a complementary profile-based view of cell types.

The combined output is suitable for planning discussions (e.g., greening/shade prioritisation), not as a direct real-time heat monitoring system.

## Limitations
- Thermal values are treated as **relative intensity indicators** (0-255 style range), not direct Celsius readings.
- Many cells can reach the upper thermal value, so interpretation is relative.
- Thermal (2012) and canopy (2021) datasets are from different years.
- Results are exploratory and should not be interpreted as causal proof of same-time canopy-temperature effects.

## Model Improvement Interpretation (No Overclaim)
The model candidate search validated the selected clustering configuration rather than producing a major performance improvement.  
Its main value is methodological justification and stability evidence across multiple scaler/k/seed settings.

## Sprint 5 Readiness / Handover Note
The notebook now includes:
- final interpretation and limitations narrative,
- threshold/sensitivity evidence,
- clustering validation summary,
- exportable handover artefacts in `outputs/`,
- and a final quality checklist section.

This supports Sprint 5: Final Testing, Handover, and Reporting.

## Suggested Presentation Flow
1. Problem and research question.
2. Data, CRS alignment, and grid-based setup.
3. Rule-based hotspots and main map outcome.
4. Sensitivity + threshold tuning and comparative analysis.
5. Clustering validation (baseline vs selected; no overclaim).
6. Final interpretation, limitations, and handover exports.
