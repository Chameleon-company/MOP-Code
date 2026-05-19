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
├── metadata.json
├── Project_3B_T126.ipynb
├── Project_3B_T126.html
└── REPORT.html
```

---

## File Guide

- `README.md`  
  Project overview, repository structure, and usage instructions

- `metadata.json`  
  Publishing metadata for the use case

- `Project_3B_T126.ipynb`  
  Main notebook demonstrating the end-to-end workflow, preprocessing, modelling, evaluation, and findings

- `Project_3B_T126.html`  
  Exported HTML version of the notebook for easier review without running Jupyter Notebook

- `REPORT.html`  
  Final project report containing detailed findings, methodology, model comparison, and conclusions

---

## Environment Setup

Recommended Python version:
- Python 3.10+

Install required packages before running the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost geopandas shapely
```

Optional packages for the LLM workflow:

```bash
pip install groq python-dotenv
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-link>
```

### 2. Open the project folder

```bash
cd Project_3B_T126
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open and run

```text
Project_3B_T126.ipynb
```

---

## Main Workflow

The notebook demonstrates:
- dataset review and selection
- data preprocessing and feature engineering
- pipe-level dataset creation
- machine learning modelling
- model evaluation and comparison
- risk scoring workflow
- Melbourne adaptation work
- LLM-based maintenance recommendation concept

---

## Team Contributions

### Dataset Review and Selection
- **Anh**: reviewed Kitchener, Netherlands, and external soil datasets
- **Joe**: reviewed Bozeman dataset
- **Shival**: reviewed Melbourne water and soil datasets
- The team selected the Kitchener dataset as the final modelling dataset

### Data Preprocessing
- **Anh**: completed Kitchener preprocessing, joins, cleaning, and pipe-level dataset creation
- **Anh and Shival**: completed Melbourne preprocessing and adaptation workflow

### Machine Learning Modelling
- **Anh**: Random Forest
- **Joe**: XGBoost
- **Shival**: Logistic Regression

### Model Evaluation and Risk Analysis
- **Anh**: aligned evaluation workflow, consolidated model results, and identified key risk drivers

### LLM Component
- **Joe and Shival**: supported LLM-based maintenance recommendation workflow

### Documentation and Final Submission
- **Team**: repository cleanup, documentation, final notebook preparation, and submission support

---

## Notes

- The notebook is intentionally structured as a concise end-to-end demonstration workflow
- The HTML exports are included for easier review without requiring Jupyter Notebook
- Large raw datasets and intermediate working files are excluded from the final submission repository where appropriate