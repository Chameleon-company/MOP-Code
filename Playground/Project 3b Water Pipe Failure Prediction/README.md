
# Water Pipe Failure Prediction

## Authors
- Rupanshi
- Ashley Mathew

**Duration:** 90 minutes
**Level:** Intermediate

---

## Overview

This project develops a machine learning system to predict the failure risk of water pipes across Melbourne's water supply network. Water pipe failures cause service disruptions, road damage, and costly emergency repairs. Most utility providers still use a reactive maintenance model, responding only after failures occur.

This project shifts that approach by using real infrastructure data and predictive modelling to identify high-risk pipes before they fail, enabling proactive maintenance and better resource allocation.

---

## Datasets

Two real-world datasets are accessed via public APIs:

| # | Dataset | Provider | API Type | Records |
|---|---|---|---|---|
| 1 | Water Supply Main Pipelines | Melbourne Water Corporation | ArcGIS REST API | 12,680 |
| 2 | Daily Rainfall (4 Major Dams) | Melbourne Water Corporation | CKAN API | 116,176 |

### Dataset 1: Melbourne Water Supply Main Pipelines

This dataset contains pipe infrastructure records with attributes including material type, diameter, length, soil condition, construction date, and service status. It is sourced from Melbourne Water Corporation via data.vic.gov.au.

### Dataset 2: Melbourne Daily Rainfall

This dataset contains daily rainfall measurements from Melbourne Water's four major harvesting storage dams (Maroondah, O'Shannassy, Thomson, and Upper Yarra), spanning from 1927 to 2020. It is sourced via the CKAN Datastore API.

---

## Methodology

### 1. Data Collection

Pipe infrastructure data is downloaded via the ArcGIS REST API in batches of 2,000 records. Rainfall data is downloaded via the CKAN API in batches of 10,000 records.

### 2. Data Cleaning

Raw data is cleaned by converting timestamps, standardising material and soil type codes, filtering unrealistic values, and handling missing data.

### 3. Target Variable Construction

Since the dataset lacks an explicit failure label, a probabilistic target variable is engineered using domain knowledge. Factors include pipe age, material risk, soil corrosivity, and operational history. Random noise is added to simulate real-world uncertainty.

### 4. Feature Engineering

Domain-specific features are created to improve model performance:

| Feature | Description |
|---|---|
| age_ratio | Pipe age divided by material lifespan |
| remaining_life | Normalised estimate of useful life remaining |
| soil_corrosivity_score | Numeric score based on soil type |
| high_risk_material | Binary flag for legacy materials (CI, MS) |
| high_risk_soil | Binary flag for corrosive soils (CLAY) |
| climate_stress | Composite score from rainfall exposure |
| cumulative_rainfall | Total rainfall during pipe's lifetime |
| extreme_rain_years | Count of above-average rainfall years |
| combined_risk | Weighted aggregate of all risk factors |

### 5. Model Training

Three ensemble models are trained and compared: XGBoost, Random Forest, and Gradient Boosting. SMOTE is applied to handle class imbalance. Standard scaling is applied to all numeric features.

### 6. Model Evaluation

Models are evaluated using accuracy, F1 score, and AUC-ROC. The best-performing model generates failure probability scores for all pipes, creating a prioritised maintenance shortlist.

### 7. AI-Assisted Maintenance Recommendations

Prompt engineering and LLM reasoning are used to translate model outputs into actionable maintenance recommendations for utility teams.

---

## Key Results

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| XGBoost | 0.7951 | 0.7255 | 0.8773 |
| Random Forest | 0.7943 | 0.7293 | 0.8741 |
| Gradient Boosting | 0.8038 | 0.7413 | 0.8816 |

**Best Model:** Gradient Boosting (AUC: 0.8816)

### Key Findings

1. Climate exposure is a strong predictor of failure. Pipes in high climate stress conditions showed a 74.8% failure rate compared to 19.4% in low stress conditions.

2. Pipe age and material composition are critical. Pipes installed before 1960 using Cast Iron (CI) and Mild Steel (MS) showed the highest failure rates.

3. The combined risk score captures multi-factorial failure. The engineered combined risk feature was the most important predictive factor.

4. Gradient Boosting achieved the best performance among the three models tested.

---

## How to Run

1. Open `Project3B.ipynb` in Google Colab
2. Click Runtime then Run all
3. The notebook will automatically download data from Melbourne Water APIs, clean and process the data, engineer features, train and evaluate three ML models, generate visualisations, and produce maintenance recommendations.

Note: An internet connection is required for API data downloads. The initial download takes approximately 2-3 minutes.

---

## Requirements

- Python 3.x
- Google Colab (recommended)
- Internet connection

### Python Packages

pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, imbalanced-learn, requests

Install all packages by running: `pip install -r requirements.txt`

---

## Project Structure

```
Project 3b Water Pipe Failure Prediction/
    Project3B.ipynb          Main notebook with all code and analysis
    README.md                This file
    requirements.txt         Python package dependencies
```

---

## Limitations

The target variable was constructed using domain knowledge and probabilistic methods rather than confirmed historical failure records. The model does not account for installation quality, maintenance history, or real-time operational conditions. The rainfall data is aggregated across four dam locations and does not capture fine-grained spatial variation across Melbourne.

---

## Data Sources

1. Melbourne Water Corporation (2024) Water Supply Main Pipelines, Available at: https://data.vic.gov.au

2. Melbourne Water Corporation (2024) Water Supply Daily Rainfall at the 4 Major Harvesting Storage Dams, Available at: https://data.vic.gov.au
