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

Several candidate datasets were reviewed during the early stage of the project to determine their suitability for supervised water pipe failure modelling. The review focused on dataset structure, data completeness, historical failure availability, scalability, and compatibility with the planned machine learning workflow.

The Melbourne Water Mains and Soil datasets were useful for the later Melbourne adaptation stage because they contained operational and infrastructure-related information relevant to local water assets. However, these datasets did not contain historical failure labels or break history records, making them unsuitable for direct supervised model training.

The Bozeman dataset provided useful examples of water main break records, but the dataset size was relatively small, containing only 158 records. This limited its suitability for reliable model development and evaluation.

The Netherlands dataset contained relatively clean and structured water main profile data together with historical break information. However, the break history was stored in a sequence-based 3D array format with shape `(10203, 6, 2)`, making it less practical for the current flat machine learning workflow using models such as Random Forest, XGBoost and Logistic Regression. Additional preprocessing and restructuring would have been required to convert the sequence data into usable tabular features.

The Kitchener Water Mains and Breaks datasets provided the strongest overall balance between dataset size, structure and modelling suitability. The datasets included 16,163 pipe asset records and 2,994 break records, with compatible identifiers (`WATMAINID` and `Related Asset ID`) that allowed historical break events to be linked directly to individual pipe assets. This structure made it possible to create a supervised pipe-level modelling dataset suitable for classification-based machine learning workflows.

One limitation of the Kitchener dataset was the lack of environmental or soil-related attributes. To address this, additional Canadian soil datasets were explored separately using GIS spatial joins to test whether environmental enrichment could improve the modelling dataset.

---

## 2.2 Final Dataset Selection

Based on the dataset review, the Kitchener Water Mains and Breaks datasets were selected as the primary datasets for historical pipe failure modelling.

The selection was primarily driven by:

- availability of both pipe asset data and historical break records
- compatible identifiers for asset-level linking
- sufficient dataset size for machine learning
- structured tabular format suitable for flat ML workflows
- strong compatibility with supervised classification approaches

The Kitchener datasets provided the most practical foundation for developing and evaluating predictive pipe failure models within the scope and timeline of the project.

---

## 2.3 External Soil Dataset Testing

To address the absence of environmental attributes in the Kitchener datasets, two external Canadian soil datasets were tested using GIS spatial joins:

1. Ontario Soil Survey Complex  
2. CANSIS / Soil Landscapes of Canada  

The objective was to enrich pipe asset locations with surrounding soil and environmental characteristics that may potentially influence water main deterioration or failure behaviour.

The spatial joins were technically successful, and soil attributes could be linked to pipe coordinates. However, most resulting classifications were relatively broad and dominated by general urban or regional categories rather than detailed pipe-level environmental characteristics. As a result, the additional soil features did not provide strong discriminatory value for the modelling stage and were not included as final modelling features.

Despite not being included in the final models, the testing process demonstrated the feasibility of future GIS-based environmental enrichment approaches for water infrastructure risk modelling.

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