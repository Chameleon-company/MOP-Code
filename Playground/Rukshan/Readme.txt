# UC00216 Weather Impact on Transport Activity Counts

## Overview

This use case explores whether weather conditions, especially rainfall, have an impact on transport activity counts in Melbourne.

The analysis combines transport activity count data with Bureau of Meteorology rainfall data to investigate whether rainfall is a useful predictor of transport usage. The project includes data loading, cleaning, feature engineering, exploratory data analysis, visualisation, model building, model evaluation, fine-tuning, and AI-style traffic explanation modules using prompt engineering and RAG-style logic.

## Use Case Scenario

As a data analyst, I want to examine whether weather conditions, especially rainfall, affect transport activity, so that I can better understand patterns in transport usage and determine whether weather is a useful predictor of demand.

## Datasets Used

### 1. Transport Activity Count Dataset

The transport activity count dataset is sourced from the City of Melbourne Open Data platform.

The official Explore API v2.1 endpoint was checked first. At the time of running the notebook, the API request returned a successful status code but zero records. Therefore, the full 2025 CSV attachment from the official dataset source was downloaded programmatically and used for the analysis.

The ZIP file contains four CSV files, which are loaded and combined into one transport activity dataset.

### 2. Weather Dataset

The weather dataset is sourced from the Bureau of Meteorology daily rainfall data for station 086338.

The Bureau of Meteorology weather CSV is included as a separate dependency file because the BOM temporary download link may expire.

The dependency file is stored in the shared repository dependency location:

```text
datascience/usecases/DEPENDENCIES/UC00216_Weather_Impact_Public_Transport/DEPENDENCIES/IDCJAC0009_086338_2025.csv
