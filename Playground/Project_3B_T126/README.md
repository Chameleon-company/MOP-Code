# Project 3B T126 - Water Pipe Failure Prediction

## Project Overview

This project develops a machine learning workflow to identify high-risk water pipes using historical water main failure data and pipe asset information.

After reviewing multiple candidate datasets, the **Kitchener water network dataset (Ontario, Canada)** was selected as the main modelling dataset. The project uses Kitchener for data preparation, feature engineering, and model comparison, with later findings intended to support adaptation to **Melbourne water infrastructure data**.

To keep the final repository clear and easy to review, the project has been organised with **one short notebook only**, while reusable preprocessing, modelling, and evaluation logic has been moved into separate Python scripts.

---

## Repository Structure

```text
Project_3B_T126/
├── README.md
├── REPORT.md
├── Project_3B.ipynb
│
├── data/
│   ├── raw/
│   │   ├── Kitchener/
│   │   ├── Melbourne/
│   │   └── external_soil/
│   └── processed/
│       ├── kitchener_breaks_clean.csv
│       ├── kitchener_mains_clean.csv
│       ├── kitchener_pipe_level.csv
│       ├── kitchener_pipe_master.csv
│       └── kitchener_model_ready.csv
│
├── src/
│   ├── kitchener_preprocessing.py
│   ├── random_forest_model.py
│   ├── logistic_regression_model.py
│   ├── xgboost_model.py
│   └── evaluation.py
│
└── archive/
    ├── old_versions/
    └── unused_files/
```

## Folder Guide

- `README.md`: project overview, structure, and contribution summary
- `REPORT.md`: short project summary, including dataset selection, preprocessing summary, and key findings
- `01_Project_3B_Demo.ipynb`: the only final notebook, used to demonstrate the workflow at a high level
- `data/`: raw, processed, and external supporting datasets
- `src/`: main reusable project code (preprocessing, modelling, evaluation, visualisation)
- `outputs/`: generated figures, saved models, and exported summaries
- `archive/`: exploratory or non-final working files not kept in the final submission

---

## Key Source Files

- `dataset_review.py`: dataset shortlisting and review support
- `kitchener_preprocessing.py`: Kitchener cleaning, joining, and pipe-level dataset creation
- `soil_spatial_review.py`: soil / spatial data usefulness testing
- `model_utils.py`: shared preprocessing and split functions
- `logistic_regression_model.py`: Logistic Regression workflow
- `random_forest_model.py`: Random Forest workflow
- `xgboost_model.py`: XGBoost workflow
- `evaluation.py`: standardised metrics and model comparison
- `visualisation.py`: reusable charts and plots

---

## Final Notebook Policy

The final submission includes only one notebook:

- `01_Project_3B_Demo.ipynb`

This notebook is intentionally kept short and is used only to:

- load the processed dataset
- import functions from `src/`
- demonstrate the modelling workflow at a high level
- show key outputs and findings

Detailed logic has been moved into `.py` files to keep the repository cleaner and easier to review.

---

## Project Workflow and Contributions

1. **Review candidate datasets**
   - **Anh**: reviewed Kitchener, Netherlands reference data, and 2 external soil datasets (Ontario Soil Survey Complex and CANSIS / Soil Landscapes of Canada (SLC))
   - **Joe**: reviewed Bozeman dataset
   - **Shival**: reviewed Melbourne water main and Melbourne soil datasets

2. **Select final modelling dataset**
   - The team selected **Kitchener** as the final modelling dataset based on suitability for supervised learning

3. **Preprocess data and create modelling datasets**
   - **Anh**: completed Kitchener mains and break preprocessing, data cleaning, joins, and pipe-level labelled dataset creation
   - **Shival**: completed Melbourne water main preprocessing for later adaptation work

4. **Train and compare machine learning models**
   - **Anh**: Random Forest
   - **Joe**: XGBoost
   - **Shival**: Logistic Regression
   - **Anh**: aligned evaluation across models using the same metrics, consolidated results, and identified key risk drivers

5. **Adapt findings to Melbourne**
   - **Anh**: applied key findings from Kitchener to support Melbourne adaptation and development of the risk identification approach

6. **LLM-based maintenance recommendation**
   - **Joe** and **Shival**: supported the LLM-based maintenance recommendation component

7. **Documentation and final repository preparation**
   - **Anh**: completed final documentation, repository cleanup, structure reorganisation, and final submission preparation

---

## Notes

- Large raw files and heavy GIS files should be stored locally or excluded via `.gitignore` where appropriate.
- Earlier exploratory or duplicate notebooks should be moved to `archive/` or excluded from the final submission branch.
- Final reviewers should be able to understand the project through:
  - `README.md`
  - `REPORT.md`
  - `01_Project_3B_Demo.ipynb`
