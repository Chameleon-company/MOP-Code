# Project 3B T126  
# Water Pipe Failure Prediction and Risk Identification

---

# Executive Summary

> *To be completed after final model evaluation and Melbourne adaptation results.*

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

> *To be completed.*

---

# 4. Model Development

> *To be completed.*

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