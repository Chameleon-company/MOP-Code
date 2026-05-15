# Project 3B T126 - Water Pipe Failure Prediction

## Project Overview

This project develops a machine learning workflow to identify high-risk water pipes using historical water main failure data and pipe asset information.

The project uses the Kitchener water network dataset (Ontario, Canada) as the primary modelling dataset, with later adaptation work using Melbourne water infrastructure data.

Main models:
- Logistic Regression
- Random Forest
- XGBoost

---

## Repository Structure

```text
Project_3B_T126/
│
├── README.md
├── REPORT.md
├── Project_3B_T126.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── dataset_review.py
│   ├── external_soil_testing.py
│   ├── kitchener_preprocessing.py
│   ├── kitchener_logistic_regression.py
│   ├── kitchener_random_forest.py
│   ├── kitchener_xgboost.py
│   ├── melbourne_preprocessing.py
│   └── model_comparison_risk_scoring.py
│
├── outputs/
├── dashboard/
└── archive/
```

---

## Folder Guide

- `README.md`: project overview and repository structure
- `REPORT.md`: final project report and key findings
- `Project_3B_T126.ipynb`: main notebook demonstrating the end-to-end workflow
- `data/`: raw and processed datasets used throughout the project
- `src/`: reusable preprocessing, modelling, and risk scoring scripts
- `outputs/`: exported figures, predictions, and generated results
- `dashboard/`: dashboard files and visual outputs
- `archive/`: exploratory, alternative, or non-final working files

---

## Environment Setup

Recommended Python version:
- Python 3.10+

Install required packages before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost geopandas shapely streamlit
```

Optional packages for dashboard and LLM workflow:

```bash
pip install groq python-dotenv
```

---

## Running the Project

1. Clone the repository

```bash
git clone <repository-link>
```

2. Open the project folder

```bash
cd Project_3B_T126
```

3. Launch Jupyter Notebook

```bash
jupyter notebook
```

4. Open and run:

```text
Project_3B_T126.ipynb
```

---

## Main Notebook

The main project workflow is demonstrated through:

- `Project_3B_T126.ipynb`

The notebook is intentionally kept concise and is mainly used to:
- load processed datasets
- import reusable functions from `src/`
- demonstrate the end-to-end modelling workflow
- present key outputs and findings

Most preprocessing, modelling, and evaluation logic has been separated into reusable `.py` scripts to keep the repository cleaner and easier to review.

---

## Project Workflow and Contributions

### Dataset Review and Selection
- **Anh**: reviewed Kitchener, Netherlands, and external soil datasets
- **Joe**: reviewed Bozeman dataset
- **Shival**: reviewed Melbourne water and soil datasets
- The team selected the Kitchener dataset as the primary modelling dataset

### Data Preprocessing
- **Anh**: completed Kitchener data cleaning, preprocessing, joins, and pipe-level dataset creation
- **Shival-Anh**: completed Melbourne preprocessing for adaptation work

### Machine Learning Modelling
- **Anh**: Random Forest
- **Joe**: XGBoost
- **Shival**: Logistic Regression

### Model Evaluation and Risk Analysis
- **Anh**: aligned model evaluation workflow, consolidated results, and identified key risk drivers

### Melbourne Adaptation and LLM Component
- **Anh**: supported Melbourne risk adaptation workflow
- **Joe** and **Shival**: supported LLM-based maintenance recommendation component

### Documentation and Repository Preparation
- **Team**: completed repository cleanup, structure reorganisation, documentation, and final submission preparation

---

## Notes

- Large raw datasets and GIS layers may be excluded from GitHub via `.gitignore` where appropriate
- Exploratory, alternative, or archived working files are stored under `archive/`
- The main project workflow and findings can be reviewed through:
  - `README.md`
  - `REPORT.md`
  - `Project_3B_T126.ipynb`