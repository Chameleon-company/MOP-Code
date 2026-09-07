# Urban Bike-Sharing Demand Prediction with Graph Learning

## Project Overview

This repository contains a Jupyter Notebook implementation of an urban bike-sharing demand prediction workflow using historical Melbourne bike-share station readings. The project models bike-share stations as a connected spatial network and applies machine learning methods to predict future bike availability, empty dock availability, and station-level operational risk.

The notebook is designed as an end-to-end academic use case. It prepares station-level time-series data, engineers temporal and graph-based features, compares baseline and machine learning models, evaluates forecasting performance, and produces decision-support outputs for bike redistribution planning.

## Main Notebook

```text
urban_bike_graph_learning.ipynb
```

This notebook contains the complete workflow, including data preparation, feature engineering, model training, graph construction, evaluation, visualisation, and final operational risk analysis.

## Project Scenario

As a transport operations analyst, the aim is to predict short-term bike availability and empty dock availability across Melbourne bike-share stations. The goal is to support evidence-based rebalancing decisions rather than relying only on manual inspection.

The project addresses three practical use cases:

1. Predict short-term bike availability at each station.
2. Predict short-term empty dock availability for bike returns.
3. Identify peak demand periods and high-pressure stations that may require operational intervention.

## Key Features

The notebook includes:

- Station-level hourly time-series preparation.
- Dynamic schema detection for flexible column handling.
- Lag and rolling feature engineering.
- Multi-horizon forecasting for 1-hour, 3-hour, and 6-hour targets.
- Non-graph baseline modelling.
- Distance-based station graph construction using NetworkX.
- Graph-derived neighbour features.
- Final 6-hour graph-aware forecasting benchmark.
- Model evaluation using MAE, RMSE, and R².
- Station-level and hour-level error analysis.
- Operational risk prioritisation for empty-bike and full-dock risks.
- Exported tables and figures for reporting.

## Dataset

The notebook expects the cleaned working dataset file to be placed in the same folder as the notebook.

Expected dataset file:

```text
station_timeseries_clean_with_features.csv
```

The original dataset source is the City of Melbourne open data page for Melbourne Bike Share station readings from 2011 to 2017.

Because the raw dataset may be large, it is recommended not to upload the full dataset directly to GitHub unless required. Instead, place the dataset locally before running the notebook.

Expected local structure:

```text
urban-bike-graph-learning/
│
├── urban_bike_graph_learning.ipynb
├── station_timeseries_clean_with_features.csv
├── README.md
└── outputs_final/
```

## Repository Structure

Recommended repository structure:

```text
urban-bike-graph-learning/
│
├── urban_bike_graph_learning.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
└── outputs_final/
    ├── tables/
    └── figures/
```

If the dataset is too large, keep it outside GitHub and explain the download or preparation steps in the `data/README.md` file.

## Installation

Create a virtual environment if preferred:

```bash
python -m venv venv
```

Activate the environment:

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not used, install the main dependencies manually:

```bash
pip install pandas numpy matplotlib scikit-learn networkx jupyter
```

## Required Libraries

The notebook uses the following main Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- networkx
- pathlib
- json
- math
- os
- warnings
- IPython

## How to Run the Notebook

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/urban-bike-graph-learning.git
```

2. Open the project folder:

```bash
cd urban-bike-graph-learning
```

3. Place the cleaned dataset in the same folder as the notebook:

```text
station_timeseries_clean_with_features.csv
```

4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open:

```text
urban_bike_graph_learning.ipynb
```

6. Run the notebook from top to bottom.

The notebook should be executed in order because later sections depend on earlier configuration, feature engineering, train-validation-test splitting, graph construction, and model outputs.

## Output Folders

The notebook automatically creates an output folder named:

```text
outputs_final/
```

Inside this folder, outputs are saved into:

```text
outputs_final/tables/
outputs_final/figures/
```

The generated outputs may include:

- Hourly demand and pressure profiles.
- Station graph statistics.
- Graph feature datasets.
- Multi-horizon model comparison results.
- Final selected model summaries.
- Actual vs predicted diagnostic plots.
- Station-level error analysis.
- Risk prediction outputs.
- Top risky station rankings.

## Main Outputs

Important output examples include:

```text
final_graph_model_comparison.csv
final_selected_model_summary.csv
final_test_predictions.csv
risk_forecast_summary.csv
detailed_risk_predictions.csv
station_graph_edges.csv
station_graph_stats.json
```

Important figure examples include:

```text
station_graph_map.png
final_graph_model_comparison_mae.png
actual_vs_predicted_bikes_6h.png
actual_vs_predicted_empty_docks_6h.png
risk_category_distribution.png
top_risky_stations.png
hourly_pressure_index.png
```

## Methodology Summary

The workflow follows these main stages:

1. Load and clean station-level bike-share data.
2. Convert readings into hourly station observations.
3. Create time-based, lag, rolling, and pressure features.
4. Generate supervised learning targets for 1-hour, 3-hour, and 6-hour forecasting.
5. Train non-graph baseline models.
6. Construct a station graph using geographical proximity.
7. Engineer graph-neighbour features based on nearby station conditions.
8. Compare graph-aware and non-graph forecasting performance.
9. Select final models using validation MAE.
10. Evaluate final models on the held-out test set.
11. Convert predictions into operational risk categories.

## Modelling Approach

The notebook compares several forecasting approaches, including:

- Naive persistence baseline.
- Station-hour historical mean baseline.
- Linear Regression.
- Random Forest Regressor.
- Graph-aware machine learning models using neighbour-derived features.

The final operational benchmark focuses on 6-hour prediction because this horizon provides enough time for bike redistribution planning.

## Evaluation Metrics

The notebook evaluates model performance using:

- MAE — Mean Absolute Error.
- RMSE — Root Mean Squared Error.
- R² — Coefficient of Determination.

The main model selection criterion is validation MAE, while final performance is reported on the held-out test set.

## Academic Use Case Validation

This project validates the three main use cases as follows:

| Use Case | Implementation | Operational Value |
|---|---|---|
| Predict bike availability | Multi-horizon bike forecasting and final 6-hour graph-aware model | Helps identify stations likely to run out of bikes |
| Predict empty dock availability | Multi-horizon empty dock forecasting and final 6-hour graph-aware model | Helps identify stations likely to become full |
| Identify high-pressure stations | Pressure index, hourly patterns, graph analysis, and risk ranking | Supports redistribution and station monitoring |

## Notes for GitHub Submission

Before uploading the repository to GitHub, check the following:

- The notebook runs from top to bottom without errors.
- The dataset path is relative, not a local absolute path such as `C:/Users/...`.
- Large raw datasets are not committed unless required.
- Generated outputs are either included selectively or regenerated by running the notebook.
- `.ipynb_checkpoints/`, cache files, and virtual environments are excluded using `.gitignore`.
- The README clearly explains how to run the notebook and where the dataset should be placed.

## Suggested `.gitignore`

A suitable `.gitignore` file for this project is:

```gitignore
# Python cache
__pycache__/
*.pyc

# Jupyter checkpoints
.ipynb_checkpoints/

# Virtual environments
venv/
.env/

# System files
.DS_Store
Thumbs.db

# Large datasets
*.csv
*.zip
*.xlsx

# Model files
*.pkl
*.joblib
*.pt
*.pth

# Temporary files
outputs_final/temp/
```

If you want to upload selected result CSV files from `outputs_final/tables`, remove `*.csv` from `.gitignore` or add exceptions for selected files.

## Author

Ronak

## References and Acknowledgements

- City of Melbourne Open Data — Melbourne Bike Share station readings 2011–2017.
- Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python.
- Hagberg, Schult, and Swart (2008) — Exploring Network Structure, Dynamics, and Function using NetworkX.
- McKinney (2010) — Data Structures for Statistical Computing in Python.
- Hunter (2007) — Matplotlib: A 2D Graphics Environment.
