# Urban Forest Health Risk Assessment

**Author:** Aidan Page | **Unit:** SIT764 Team Project A | **Trimester:** T1 2026  
**Client:** City of Melbourne — Parks & Urban Forestry

---

## Overview

Pipeline built for SIT764 that predicts which of Melbourne's ~82,000 trees are at health risk, using tree census data, IoT microclimate sensors, BOM weather records, and soil moisture sensors. The model assigns each tree a risk class (LOW / MEDIUM / HIGH) based on tree characteristics and environmental conditions.

Layer 3 (RAG/LLM arborist report generation) is planned for the following trimester.

## Pipeline

The entire pipeline runs in a single notebook: `urban_forest_use_case.ipynb`

```
DEPENDENCIES/ (raw data)
       ↓
1. Data Loading
2. Exploratory Data Analysis
3. Data Cleaning
4. Spatial Joins  (trees → nearest microclimate sensor, nearest soil sensor)
5. Feature Engineering  (rolling weather windows, heatwave flags, sensor aggregates)
6. Machine Learning  (Logistic Regression baseline → XGBoost + SHAP)
```

## Setup

**1. Clone the repo and navigate to this folder**

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install pandas geopandas matplotlib seaborn scikit-learn xgboost shap
```

Note: XGBoost requires `libomp` on macOS — install with `brew install libomp` if you get an import error.

**4. Add raw data files to the `DEPENDENCIES/` folder**

All files need to be downloaded manually and placed in `DEPENDENCIES/` before running the notebook.

| File | Source |
|------|--------|
| `trees-with-species-and-dimensions-urban-forest.geojson` | [City of Melbourne Open Data](https://data.melbourne.vic.gov.au) |
| `microclimate-sensors-data.csv` | [City of Melbourne Open Data](https://data.melbourne.vic.gov.au) |
| `IDCJAC0010_086338_1800_Data.csv` | [BOM Climate Data](http://www.bom.gov.au/climate/data/) — Station 086338, daily max temp |
| `IDCJAC0009_086338_1800_Data.csv` | [BOM Climate Data](http://www.bom.gov.au/climate/data/) — Station 086338, daily rainfall |
| `soil-sensor-locations.csv` | [City of Melbourne Open Data](https://data.melbourne.vic.gov.au) |
| `soil-sensor-readings-historical-data.csv` | [City of Melbourne Open Data](https://data.melbourne.vic.gov.au) |

**5. Run the notebook**

Open `urban_forest_use_case.ipynb` in VS Code (or Jupyter) and run all cells top to bottom.

## Key design decisions

- **Target variable:** derived from CoM's `useful_life_expectency_value` field — ≤20 years remaining = HIGH risk, 30 = MEDIUM, ≥40 = LOW. Not a perfect proxy but the best available label in this dataset without a separate inspection history.
- **CRS:** spatial joins use EPSG:7855 (GDA2020 / MGA zone 55) — projected CRS with metre units for Melbourne. Had to reproject from the raw geographic coordinates before running `sjoin_nearest`, otherwise distances come out in degrees not metres.
- **Class imbalance:** HIGH risk trees are only ~6% of the dataset. Addressed using `compute_sample_weight('balanced')` passed into XGBoost's fit — improved HIGH recall from ~15% to ~61%.
- **No intermediate files:** everything runs in memory from `DEPENDENCIES/` through to model output. No processed CSVs saved to disk between steps.

## Results

| Model | Test Macro F1 |
|-------|--------------|
| Logistic Regression (baseline) | see notebook output |
| XGBoost | see notebook output |
