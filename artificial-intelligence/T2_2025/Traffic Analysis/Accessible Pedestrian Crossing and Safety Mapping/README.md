# Pedestrian Crash Risk Analysis & Prediction

## 📌 Project Overview

This project analyses and predicts **pedestrian crash risk in Victoria** using road environment factors. The goal is to identify high-risk conditions (e.g., lighting, speed zones, road geometry) and provide interpretable machine learning models that support safer and more sustainable urban planning.

The work forms part of **Use Case 1 Traffic Analysis (T2_2025)** in the MOP AI+IoT capstone project, contributing to SDG 11 (Sustainable Cities).

---

## 🔧 Pipeline Summary

1. **Data Preprocessing**

   * Cleaned and filtered the Victorian Road Crash dataset
   * Performed feature engineering (categorical encoding, scaling, correlation-based selection)
   * Excluded post-crash outcome variables (e.g., fatalities) to ensure deployment feasibility

2. **Exploratory Data Analysis (EDA)**

   * Generated distribution and correlation plots
   * Analysed crash frequency by speed zone, lighting, and road type

3. **Model Training & Evaluation**

   * Implemented classifiers: Logistic Regression, SVM, Decision Tree, KNN, Random Forest, XGBoost, Neural Network
   * Integrated the top-performing classifiers (Decision Tree + XGBoost) into a **Voting Classifier**
   * Achieved **97.92% accuracy** after redefining risk into binary classes (low vs high)

4. **Interpretability & Insights**

   * Produced feature importance and permutation importance plots
   * Identified key predictors of pedestrian crash risk

5. **Deployment & Visualisation**

   * Built a **Streamlit dashboard** to showcase results interactively
   * Provided recommendations for real-world deployment

---

## 📊 Key Results

* **Voting Classifier Accuracy**: 97.92%
* **Target Refinement**: Binary classification improved interpretability and model performance
* **Stakeholder Recognition**: Product owner described the **98% traffic analysis accuracy** as “very impressive,” highlighting real-world applicability

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Chameleon-company/MOP-Code.git
cd "MOP-Code/artificial-intelligence/T2_2025/Traffic Analysis/Accessible Pedestrian Crossing and Safety Mapping"

```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:

* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `scikit-learn`, `xgboost`
* `streamlit`, `jupyter`

### 3. Run the notebook

Execute cells in the notebook to reproduce preprocessing, EDA, and model training.

### 4. (Optional) Launch the dashboard

```bash
streamlit run dashboard_risk.py
```

---

## 📂 Repository Contents

* `Use_case_1_Pedestrian_Crash_Risk_Analysis_&_Prediction.ipynb` – Full pipeline (preprocessing, EDA, modelling, evaluation)
* `dashboard_risk.py` – Streamlit app for interactive visualisation (if included)
* `requirements.txt` – Dependencies for reproducing results
* `README.md` – Project documentation (this file)
* `Pedestrian_Crash_Risk_Analysis_&_Prediction (1).ipynb` – Early EDA and data cleaning (Sprint 1–2). *For reference only, not part of final pipeline.*

---

## 👥 Contributors

* **Project Manager / Use Case 1 Lead**: KAH YAN LAI
* Team Members: EMIRA SHEIKH, ROHAN KORLAHALLI, SONA JOSEPH, SANDUNI CHAMINDI THATHSARANI DENAGAMA GAM ACHARIGE

---

## 📌 Future Work

* Extend dataset integration (e.g., pedestrian counts, weather, and environmental data)
* Optimise deployment for real-time predictions
* Enhance dashboard with additional stakeholder-facing features

---



