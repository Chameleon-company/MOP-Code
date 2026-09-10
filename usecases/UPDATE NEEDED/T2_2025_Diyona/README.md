# UC00211 – Predicting Pedestrian Congestion Using Time Series Analysis

## Overview

This use case focuses on analysing and predicting pedestrian congestion patterns using historical pedestrian counting data from the Melbourne Open Data platform. The objective is to understand pedestrian movement trends and develop a simple time-series prediction model to forecast pedestrian counts.

## Dataset

Dataset used:
- Pedestrian Counting System (counts per hour)
- Source: Melbourne Open Data (MOP)
- Link: https://data.melbourne.vic.gov.au/

Key columns used:
- sensing_date → Date of observation
- hourday → Hour of observation
- location_id → Sensor location identifier
- sensor_name → Sensor location name
- pedestriancount → Total pedestrian count

## Methodology

The workflow followed these main steps:

### Data Preparation
- Loaded dataset using API/CSV
- Selected relevant columns
- Converted date to datetime format
- Sorted data for time series analysis

### Exploratory Data Analysis
- Analysed pedestrian trends over time
- Identified peak and low traffic hours
- Visualised hourly pedestrian patterns

### Feature Engineering
Created additional features including:
- Day of week
- Weekend indicator
- Previous hour pedestrian count (lag feature)
- Rolling average pedestrian count

### Modelling
Two models were implemented:

Baseline model:
- Used previous hour pedestrian count as prediction

Regression model:
- Used temporal features and lag values
- Compared performance with baseline

### Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Congestion Classification
Pedestrian counts were categorized into:
- Low congestion
- Medium congestion
- High congestion

## Results

The regression model showed improved prediction performance compared to the baseline model. Results indicate pedestrian flow follows strong daily temporal patterns, particularly during daytime business hours.

## Key Insights

- Peak pedestrian traffic occurs during midday hours
- Lowest traffic occurs during early morning
- Pedestrian flow follows predictable daily cycles
- Temporal features improve prediction accuracy

## Future Improvements

Potential future improvements include:

- Testing advanced time series models (SARIMA)
- Using Random Forest or Gradient Boosting
- Adding weekly lag features
- Predicting future congestion periods
- Multi-sensor prediction modelling

## Files Included

- Predicting_Pedestrian_Congestion_Using_Time_Series_Analysis.ipynb → Main notebook
- README.md → Project description

## Author

Diyona Robert  
Personal Data Science Use Case
