UC00216 Weather Impact on Transport Activity Counts

Author: Rukshan Dias
Student ID: s224326349

Overview

This use case explores whether weather conditions, especially rainfall, have an impact on transport activity counts in Melbourne.

The analysis combines transport activity count data with Bureau of Meteorology rainfall data to investigate whether rainfall is a useful predictor of transport usage. The project includes data loading, cleaning, feature engineering, exploratory data analysis, visualisation, model building, model evaluation, and an AI-style traffic explanation module.

Use Case Scenario

As a data analyst, I want to examine whether weather conditions, especially rainfall, affect transport activity, so that I can better understand patterns in transport usage and determine whether weather is a useful predictor of demand.

Datasets Used

1. Transport Activity Count Dataset

The transport activity count dataset is sourced from the City of Melbourne Open Data platform.

The official Explore API v2.1 endpoint was checked first. At the time of running the notebook, the API request returned a successful status code but zero records. Therefore, the full 2025 CSV attachment from the official dataset source was downloaded programmatically and used for the analysis.

The ZIP file contains four CSV files, which are combined into one transport activity dataset.

2. Weather Dataset

The weather dataset is sourced from the Bureau of Meteorology daily rainfall data for station 086338.

The notebook first attempts to download the BOM ZIP file programmatically. Because the BOM temporary download link may expire, a local fallback CSV is also stored inside the DEPENDENCIES folder to keep the use case reproducible.

Main Steps Covered

1. Load the transport activity dataset
2. Check the official City of Melbourne API v2.1 endpoint
3. Load and combine four transport CSV files
4. Load the BOM rainfall dataset
5. Clean both datasets
6. Create time-based and weather-based features
7. Merge datasets using the date field
8. Perform exploratory data analysis
9. Visualise rainfall and transport usage patterns
10. Compare rainy and non-rainy day activity
11. Build a Linear Regression model
12. Evaluate model performance using MAE, MSE, RMSE, and R2
13. Compare performance with a Random Forest Regressor
14. Interpret model coefficients and correlations
15. Implement an LLM-style traffic explanation module
16. Implement a prompt engineering and RAG-based explanation module

Key Findings

- Rainfall has a very weak relationship with transport activity counts.
- Average transport usage is similar on rainy and non-rainy days.
- Time of day has a stronger influence on transport activity than rainfall.
- Transport activity is lowest during early morning hours and increases during typical commuting periods.
- The Linear Regression model has low predictive performance because rainfall and hour alone are not enough to explain transport activity.
- A Random Forest model slightly improves performance, but the improvement is still limited.
- Additional features such as location, day of week, holidays, events, and transport mode would likely improve prediction accuracy.

Machine Learning Models Used

1. Linear Regression

A Linear Regression model was used as the baseline model to predict transport usage using rainfall and hour of day.

2. Random Forest Regressor

A Random Forest Regressor was tested as an alternative model to check whether a non-linear model could improve prediction performance.

AI Explanation Modules

1. LLM-Based Traffic Explanation Module

This module allows the user to type a natural-language traffic or weather question. The system extracts useful information such as time, rainfall condition, and location, then generates a human-readable explanation based on the trained model and rule-based reasoning.

Example question:

Will transport usage be busy near Melbourne CBD at 5PM if it is raining?

2. Prompt Engineering and RAG-Based Traffic Explanation Module

This module demonstrates prompt engineering and Retrieval-Augmented Generation-style logic. It retrieves analytical context from the dataset and model outputs, builds a structured prompt, and generates an AI-style explanation.

Example question:

What is the best time to travel near Southbank if I want to avoid peak transport activity?

Requirements

The notebook uses the following Python libraries:

- requests
- zipfile
- io
- os
- re
- numpy
- pandas
- matplotlib
- sklearn

How to Run

1. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells from top to bottom.
3. Make sure the DEPENDENCIES folder is included in the same directory as the notebook.
4. If the BOM temporary URL fails, the notebook will load the fallback CSV from the DEPENDENCIES folder.
5. Review the visualisations, model results, and AI explanation module outputs.

Notes for Reviewers

- The City of Melbourne API v2.1 endpoint was checked in the notebook.
- The endpoint returned zero records at the time of execution, so the official CSV attachment was used as the working dataset.
- The transport ZIP file contains four CSV files, all of which are loaded and combined.
- The BOM rainfall file is loaded programmatically, with a fallback CSV stored in DEPENDENCIES.
- No API keys are used or exposed.
- Australian English spelling is used throughout the notebook.
- Visualisations include labels, titles, and written interpretations.
- The use case follows a step-by-step tutorial structure.

Conclusion

This use case shows that rainfall has minimal impact on transport activity counts, while time of day plays a more meaningful role. The machine learning results confirm that rainfall and hour alone are not strong enough predictors for accurate transport usage prediction. The AI-style explanation modules improve interpretability by converting analytical results into clear, human-readable recommendations.