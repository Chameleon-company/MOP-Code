\# README.md



```markdown



# Melbourne Traffic Accident Risk Prediction

## Project Overview

This project aims to analyse and predict traffic accident risk in Melbourne using historical crash data, weather data, and mobility-related indicators.

By combining crash records with environmental and active transport features, the project identifies patterns associated with accident severity and road safety risk. The work supports evidence-based decision-making for local government and road safety stakeholders.

The final project output includes cleaned and integrated datasets, exploratory analysis, machine learning models, model evaluation, feature importance analysis, and a final dashboard-style project summary notebook.

---

## Project Objectives

The primary objectives of this project are:

- Analyse historical crash patterns across time, severity, speed zones, and environmental conditions
- Examine the relationship between rainfall, weather-related variables, and crash records
- Incorporate pedestrian and bicycle density features into crash severity modelling
- Develop and compare machine learning models for crash severity prediction
- Evaluate model performance using accuracy, macro/weighted metrics, and class-wise results
- Identify important predictive features using Random Forest feature importance
- Create a dashboard-style summary notebook to communicate insights to non-technical stakeholders

---

## Team Members

- Suba Thinakaran
- Khalid Ameen
- Burhanuddin Ujjainwala

---

## Datasets Used

### 1. Victorian Road Crash Data

Source: Victoria Government Open Data

Contains historical traffic crash records including:

- crash date and time
- location information
- crash severity
- road characteristics
- speed zones

---

### 2. Melbourne Weather Data

Source: Bureau of Meteorology / Melbourne weather data sources

Weather variables include:

- rainfall
- minimum and maximum temperature
- humidity
- wind speed

Weather data was integrated to understand how environmental conditions may contribute to crash patterns and severity.

---

### 3. Melbourne Pedestrian Counts

Source: City of Melbourne Open Data Portal / MOP data

The pedestrian dataset contains:

- pedestrian activity measurements
- sensor-based observations
- timestamped movement data

Pedestrian activity was used to create mobility-related indicators for crash risk modelling.

---

### 4. Melbourne Bicycle Counts

Source: City of Melbourne Open Data Portal / MOP data

The bicycle dataset includes:

- cyclist activity measurements
- timestamps
- location or sensor-based movement data

Bicycle activity was used together with pedestrian data to create mobility density features.

---

## Project Workflow

The project follows a complete data science workflow:

1. Data collection and source review
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Dataset integration
6. Machine learning modelling
7. Model comparison and evaluation
8. Class-wise crash severity performance review
9. Feature importance analysis
10. Dashboard-style project summary and stakeholder recommendations

---

## Key Analysis Areas

The project summary notebook includes the following analysis sections:

- KPI summary of the final integrated dataset
- Crash severity distribution
- Crash trend over time
- Crash pattern by hour of day
- Speed zone and severity analysis
- Rainfall and crash severity insights
- Pedestrian and bicycle mobility density analysis
- Machine learning model comparison
- Class-wise crash severity model performance
- Random Forest feature importance
- Final model summary
- Stakeholder findings, recommendations, limitations, and conclusion

---

## Modelling Summary

Several machine learning models were compared, including:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Based on the final model comparison, **Random Forest** achieved the strongest overall performance across the available metrics, including accuracy, macro F1-score, and weighted F1-score.

However, class-wise results showed that fatal crash prediction remains challenging due to class imbalance. Therefore, the model should be treated as a **decision-support tool** rather than a fully operational crash prediction system.

---

## Key Findings

The final analysis found that:

- Crash records are not evenly distributed across time, severity, speed zones, and environmental conditions.
- Crash activity increases during daytime and commuting periods, especially around the afternoon peak.
- The dataset is dominated by injury-related crashes, while fatal crashes are rare but highly important.
- 50–60 km/h speed zones contain a high number of crash records, likely reflecting urban and suburban road environments.
- Higher-speed zones such as 80–100 km/h remain important for serious and fatal crash risk review.
- Rainfall analysis must be interpreted with exposure in mind, because most crashes occurred during no-rain conditions.
- Pedestrian and bicycle density features provide useful context for understanding crash severity patterns.
- Speed zone, weather variables, mobility density, and time-based features were important contributors to the Random Forest model.

---

## Final Project Summary Notebook

The final dashboard-style project summary notebook is intended as the main handover notebook for stakeholders.

It consolidates key outputs from the earlier technical notebooks and presents them in a more readable format.

Notebook name: 19_TARP_Final_Project_Summary_Notebook.ipynb



