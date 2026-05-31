# UC00215 – Crash Traffic Prediction System

## Project Overview
This project focuses on predicting traffic congestion levels using historical traffic and crash-related data. The system applies machine learning techniques to analyse temporal traffic patterns and generate congestion predictions based on time-related features such as hour of the day, weekdays, weekends, and peak-hour conditions.

The project further integrates Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Prompt Engineering to generate human-readable explanations for predicted congestion levels, improving interpretability and decision support.

---

## Objectives
- Analyse historical traffic congestion patterns
- Perform data preprocessing and feature engineering
- Build and evaluate machine learning prediction models
- Integrate crash-related contextual information
- Generate explainable AI outputs using LLMs and RAG
- Support intelligent urban traffic management insights

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Gradient Boosting
- Random Forest
- Phi3-mini LLM
- Ollama
- Jupyter Notebook

---

## Project Workflow

### 1. Data Preprocessing
- Cleaned missing and duplicate records
- Converted timestamps into hourly traffic patterns
- Generated engineered features:
  - Day of week
  - Weekend indicator
  - Peak-hour indicator
  - Cyclical hour encoding (sin/cos)

### 2. Exploratory Data Analysis
- Analysed hourly congestion distributions
- Visualised peak traffic periods
- Investigated crash frequency patterns

### 3. Machine Learning Modelling
Models implemented:
- Logistic Regression
- Random Forest
- Gradient Boosting

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 4. Explainable AI Integration
The project integrates:
- Phi3-mini LLM
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)

The LLM uses retrieved historical traffic context to explain:
- Predicted congestion level
- Traffic intensity
- Historical crash frequency
- Possible traffic causes

---

## Current Results
- Improved congestion prediction performance using Gradient Boosting
- Successfully integrated Phi3-mini LLM explanation pipeline
- Implemented RAG-based contextual retrieval
- Generated interpretable congestion explanations for end users

---

## Future Improvements
- Integrate real-time traffic APIs
- Add weather and event datasets
- Improve RAG retrieval quality
- Deploy as a web-based dashboard
- Enhance LLM explanation accuracy

---

## Repository Structure

```plaintext
Playground/
└── tiendathoang/
    ├── UC00215 - Crash Traffic Prediction System.ipynb
    ├── UC00215 - Crash Traffic Prediction System - Official.ipynb
    └── README.md