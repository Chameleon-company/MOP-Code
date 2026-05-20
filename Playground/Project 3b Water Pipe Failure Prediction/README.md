# Water Pipe Failure Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--Learn_|_XGBoost-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)

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

This dataset contains daily rainfall measurements from Melbourne Water's four major harvesting storage dams (Maroondah, O'Shannassy, Thomson, and Upper Yarra), spanning from 1892 to 2020. It is sourced via the CKAN Datastore API.

---

## Methodology

### 1. Data Collection

Pipe infrastructure data is downloaded via the ArcGIS REST API in batches of 2,000 records. Rainfall data is downloaded via the CKAN API in batches of 10,000 records.

### 2. Data Cleaning

Raw data is cleaned by converting timestamps to years, standardising material and soil type codes, filtering out unrealistic values (e.g., pipe age, dimensions), and handling missing data.

### 3. Target Variable Construction

Since the dataset lacks an explicit failure label, a **probabilistic target variable** is engineered using domain knowledge. A `failure_prob` score is calculated for each pipe based on factors like age, material, soil type, and operational history. Random noise is added to simulate real-world uncertainty. Pipes with a score above the **65th percentile** are labelled as `failed` (at-risk).

### 4. Feature Engineering

Domain-specific features are created to improve model performance. Key engineered features include:

| Feature | Description |
|---|---|
| `age_ratio` | Pipe age divided by its material's expected lifespan. |
| `remaining_life` | A normalised estimate of the pipe's useful life remaining. |
| `climate_stress` | A composite score reflecting cumulative and extreme rainfall exposure. |
| `soil_corrosivity_score` | A numeric score quantifying the corrosive potential of the soil type. |
| `high_risk_material` | A binary flag for legacy materials known to be at higher risk (e.g., CI, MS). |
| `combined_risk` | A weighted aggregate of all major risk factors into a single, powerful feature. |

Other features like log-transformed dimensions (`diameter_log`, `length_log`) were also created to normalise distributions.

### 5. Model Training

Three ensemble models are trained and compared: XGBoost, Random Forest, and Gradient Boosting. To handle class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to the training data. All numeric features are standardised using `StandardScaler`.

### 6. Model Evaluation

Models are evaluated using accuracy, F1 score, and AUC-ROC. The best-performing model is then used to generate failure probability scores for all pipes, creating a prioritised maintenance shortlist.

### 7. AI-Assisted Maintenance Recommendations

Prompt engineering and LLM reasoning are used to translate the model's technical outputs into actionable, plain-language maintenance recommendations for utility teams.

---

## Key Results

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| XGBoost | 0.7951 | 0.7255 | 0.8773 |
| Random Forest | 0.7943 | 0.7293 | 0.8741 |
| **Gradient Boosting** | **0.8038** | **0.7413** | **0.8816** |

**Best Model:** Gradient Boosting (AUC: 0.8816)

### Key Findings

1.  **Climate Exposure is a Strong Predictor:** Pipes in the highest quintile of climate stress showed a **74.8%** failure rate, compared to just **19.4%** for those in the lowest quintile.

2.  **Age and Material Composition are Critical:** Pipes installed before 1960 using legacy materials like Cast Iron (CI) and Mild Steel (MS) showed the highest failure rates, confirming their vulnerability.

3.  **The `combined_risk` Score Captures Multi-Factor Failure:** The engineered `combined_risk` feature was identified by the model as the most important predictive factor, proving that failure is driven by an interaction of variables.

4.  **Gradient Boosting Achieved the Best Performance** among the three models tested, providing the best balance of accuracy and discriminative ability.

---

## How to Run

1.  Open `Project3B.ipynb` in Google Colab.
2.  Click `Runtime` then `Run all`.
3.  The notebook will automatically download data from Melbourne Water APIs, clean and process the data, engineer features, train and evaluate the three ML models, generate visualisations, and produce maintenance recommendations.

*Note: An internet connection is required for API data downloads. The initial download takes approximately 2-3 minutes.*

---

## Requirements

- Python 3.x
- Google Colab (recommended)
- Internet connection

### Python Packages

`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `requests`

Install all packages by running:
```bash
pip install -r requirements.txt
