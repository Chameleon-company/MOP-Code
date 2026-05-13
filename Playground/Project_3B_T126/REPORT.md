# Project 3B T126  
# Water Pipe Failure Prediction and Risk Identification

---

# Executive Summary

This project developed an interactive web-based system support system for Water Pipe Failure Prediction Analysis to predict and prioritize water main pipe failures. Using historical data from Kitchener (Canada) and rich asset data from Melbourne, we built a risk scoring framework and a user-friendly Streamlit dashboard. 

The final solution provides: 

- An accurate risk classification of High, Medium and Low 

- The interactive visualizations for maintenance planning 

- AI-powered (LLM) maintenance recommendations 

- A clean, scalable dashboard ready for operational use 

The system helps maintenance teams move from reactive to proactive asset management strategy, potentially reducing unplanned outages and optimizing replacement budgets.  

---

# 1. Project Overview

## 1.1 Introduction

Water main failures can cause significant operational disruption, repair costs, water loss and public safety risks. As water infrastructure continues to age, there is increasing interest in using data-driven approaches to support proactive maintenance and risk identification.

The aim of this project is to develop a machine learning-based workflow for predicting water pipe failure risk using historical water main break data. The project focuses on identifying patterns and key risk drivers associated with pipe failures, then adapting those findings to Melbourne water main data to support risk identification and maintenance planning.

Three supervised machine learning models were developed and compared using historical overseas pipe failure datasets. The final project also explores how Large Language Models (LLMs) can be used to generate maintenance recommendations based on identified risk conditions.

---

## 1.2 Project Objectives

The main objectives of the project are:

- review and assess available water pipe datasets
- identify a suitable dataset for supervised pipe failure modelling
- preprocess and engineer pipe-level modelling features
- develop and compare multiple machine learning models
- identify key pipe failure risk drivers
- adapt modelling findings to Melbourne water main data
- build a simple risk identification workflow
- explore LLM-based maintenance recommendation generation

---

# 2. Dataset Review and Selection

## 2.1 Candidate Dataset Review

Several candidate datasets were reviewed during the early stage of the project to assess their suitability for supervised water pipe failure modelling. The review focused on dataset scale, data structure, historical failure availability, feature completeness, and compatibility with the planned machine learning workflow.

| Dataset | Outcome |
|---|---|
| Melbourne Water Mains + Soil | Useful for later adaptation, but no historical failure labels for direct supervised modelling |
| Bozeman | Small dataset with limited records, not strong enough as the main modelling dataset |
| Netherlands reference data | Useful as reference material, but less suitable for our current flat ML workflow |
| Kitchener Water Mains + Breaks | Selected as the final modelling dataset because it includes both pipe asset data and historical break records, even though it does not include soil or environmental factors |
| External Canadian soil datasets | Tested as supporting data, but not used as main modelling features |

The Melbourne Water Mains and Soil datasets contained 12,680 water main records and more than 900,000 soil monitoring records. These datasets were considered useful for the later Melbourne adaptation and risk identification stages because they contained operational and infrastructure-related attributes relevant to local water assets. However, the datasets did not include historical break labels or failure records required for supervised model training.

The Bozeman dataset provided historical water main break records but contained only 158 records in total. Due to its limited size, it was not considered suitable for reliable model development or comparison across multiple machine learning approaches.

The Netherlands dataset contained 10,203 water main records together with environmental-related features. However, the historical break data was stored in a sequence-based 3D array format with shape `(10203, 6, 2)` rather than a conventional tabular structure. While the dataset was relatively complete, the sequence-based format was less compatible with the current flat machine learning workflow using models such as Random Forest, XGBoost and Logistic Regression.

The Kitchener Water Mains and Breaks datasets provided the strongest balance between dataset scale, structure and modelling suitability. The datasets contained 16,163 water main asset records and 2,994 break records, with compatible identifiers (`WATMAINID` and `Related Asset ID`) that allowed historical break events to be linked directly to individual pipe assets. This structure supported the creation of a supervised pipe-level modelling dataset suitable for classification-based machine learning workflows.

Based on the overall review, the Kitchener datasets were selected as the primary datasets for historical pipe failure modelling.

---

## 2.2 External Soil Dataset Testing

To address the absence of environmental attributes in the Kitchener datasets, two external Canadian soil datasets were tested using GIS spatial joins:

1. Ontario Soil Survey Complex (polygon-based soil survey data)  
2. CANSIS / Soil Landscapes of Canada (national soil landscape polygons and soil component tables)

The objective was to enrich Kitchener break locations with surrounding soil characteristics that may potentially influence water main deterioration or failure behaviour.

Both spatial joins achieved complete geographic coverage, with all 2,994 break records successfully matched to soil polygons. However, the Ontario Soil Survey Complex results were heavily dominated by broad urban classifications, with 2,943 records labelled as `URBAN` and 98.53% of texture values recorded as `NA`. This provided limited useful soil variation for modelling.

The CANSIS dataset produced cleaner and more complete soil attributes, but the results still showed limited diversity. Most break locations were concentrated in only three soil classes (`FOX`, `BURFORD`, and `WATERLOO`), while drainage and texture values were nearly constant across the dataset.

As a result, the external soil datasets did not provide sufficient pipe-level discriminatory value and were not included as final modelling features. Nevertheless, the testing process demonstrated the feasibility of future GIS-based environmental enrichment workflows for water infrastructure risk modelling.

---

# 3. Kitchener Data Preprocessing

The Kitchener preprocessing stage converted the raw water mains and water main break datasets into a clean pipe-level dataset for supervised modelling.

The raw data contained 16,163 water main asset records and 2,994 break records. Initial cleaning removed weak fields with high missingness and operational fields that were not required for the modelling workflow. After cleaning, the main dataset was reduced from 27 to 22 columns, while the breaks dataset was reduced from 37 to 18 columns.

The break data was originally event-level, while the main data was asset-level. To create a supervised modelling dataset, break records were linked back to pipe assets using `Related Asset ID` from the break dataset and `WATMAINID` from the mains dataset. This matching process linked 2,452 break records to water main assets, giving an asset match rate of 81.9%. The data was then filtered to MAIN break records only, resulting in 2,451 matched MAIN break records for pipe-level aggregation.

Break history was aggregated so that each pipe was represented by one row. For each pipe, the preprocessing created fields such as break count, first break date, last break date and the binary target variable `has_break`. Pipes with at least one matched historical MAIN break were labelled as `has_break = 1`; all other pipes were labelled as `has_break = 0`.

The final pipe-level dataset contained 16,163 records and 26 columns. A separate pipe master dataset with traceability fields contained 28 columns, while the final model-ready dataset contained 16,163 records and 20 columns after removing identifiers, date fields and leakage-related fields.

The final target distribution was highly imbalanced. A total of 14,929 pipes, or 92.37%, had no observed historical break, while 1,234 pipes, or 7.63%, had at least one observed break. This class imbalance was considered during model development through stratified splitting and evaluation metrics such as precision, recall, F1-score, ROC-AUC and PR-AUC.

---

# 4. Model Development

## 4.1 Random Forest Model

The Random Forest model was trained using the processed Kitchener model-ready dataset with a consistent 70/15/15 train, validation and test split. The split produced 11,314 training records, 2,424 validation records and 2,425 test records.

The model performed strongly across both validation and test sets. On the validation set, it achieved ROC-AUC of 0.9910, PR-AUC of 0.9678, precision of 0.9713, recall of 0.9135 and F1-score of 0.9415. On the held-out test set, the model achieved ROC-AUC of 0.9906, PR-AUC of 0.9596, precision of 0.9508, recall of 0.9405 and F1-score of 0.9457.

The test confusion matrix showed 2,231 true negatives and 174 true positives, with only 9 false positives and 11 false negatives. This indicates that the model performed well in identifying pipes with observed historical breaks while maintaining a low number of incorrect high-risk predictions.

The feature importance results showed that `condition_score` was the most influential predictor, with an importance score of 0.4660. This was followed by `shape__length` at 0.1642 and `pipe_age` at 0.1013. Material-related features also contributed to prediction, particularly cast iron (`material_CI`) and PVC (`material_PVC`). Other contributing variables included pipe size, cleaning area, criticality and pressure zone.

Overall, the Random Forest model provided strong predictive performance and useful interpretability. The results suggest that pipe condition, pipe length, pipe age and material were key indicators of historical break risk in the Kitchener dataset. These findings are useful for the later Melbourne adaptation stage, where similar asset-level features can be used to support risk identification.

## 4.2 XGBoost Model

The XGBoost model was trained using the same processed Kitchener model-ready dataset and the same 70/15/15 train, validation and test split used for the other models. The split produced 11,314 training records, 2,424 validation records and 2,425 test records.

The model used the best hyperparameters identified during earlier tuning, rather than rerunning Optuna in the final demonstration notebook. This kept the final workflow lightweight and reproducible. Class imbalance was handled using `scale_pos_weight`, calculated from the training set as 12.09.

On the held-out test set, XGBoost achieved ROC-AUC of 0.9899, PR-AUC of 0.9614, precision of 0.9133, recall of 0.9676 and F1-score of 0.9396. The confusion matrix showed 2,223 true negatives, 179 true positives, 17 false positives and 6 false negatives.

These results show that XGBoost performed very strongly, particularly in recall. It missed only 6 of the 185 break cases in the test set, making it effective for identifying high-risk pipes. However, it produced more false positives than Random Forest, meaning it was slightly more aggressive in classifying pipes as break-risk.

The feature importance results showed that `condition_score` was the most influential predictor, followed by `material`, `shape__length`, `pipe_age`, `rel_cleaning_subarea`, `pressure_zone` and `rel_cleaning_area`. This aligns well with the Random Forest findings and reinforces that pipe condition, material, length, age and operational network context are important drivers of predicted failure risk.

Overall, XGBoost provided strong predictive performance and high break-case detection. It is useful for model comparison because it offers a different modelling approach from Random Forest while still identifying similar key risk drivers.

---

# 5. Model Comparison and Evaluation

> *To be completed.*

---

# 6. Melbourne Water Data Adaptation

> *To be completed.*

---

# 7. LLM Recommendation Workflow

> *To be completed.*

---

# 8. Discussion and Limitations

> *To be completed.*

---

# 9. Conclusion

> *To be completed.*

---

# References

> *To be completed.*