# Population-Level Sleep Efficiency Prediction using Sleep Stages & Demographics

## Project Summary

This project aims to predict **sleep efficiency** (nsrr_ttleffsp_f1) using a combination of **sleep stage sequences** and **demographic features**. It forms part of the **MOP Capstone – Health Behaviour Analysis (AI+IoT stream)**, addressing **SDG 3: Good Health and Wellbeing**.

The workflow includes processing XML-based hypnogram data, extracting sleep stage sequences, merging them with cleaned and encoded demographic features, and training an LSTM-based deep learning model. The final model outputs a regression prediction of sleep efficiency.



## Pipeline Overview

### Data Preprocessing

- **Demographic Data:**
  - Extracted from NSRR datasets.
  - Selected features: `nsrr_age`, `nsrr_sex`, `nsrr_bmi`, `nsrr_race`, `nsrr_current_smoker`, `nsrr_bp_systolic`, `nsrr_bp_diastolic`.
  - Categorical variables encoded using `.map()` and `pd.get_dummies()`.
  - Numeric features scaled using `MinMaxScaler`.
  - Target (`nsrr_ttleffsp_f1`) also normalized.

- **Sleep Stage Sequences:**
  - Built from event-level XML files using a custom `build_hypnogram()` function.
  - Each sequence encoded to map:
    - `Wake` → 0  
    - `N1` → 1  
    - `N2` → 2  
    - `N3` → 3  
    - `REM` → 4
  - Final sequences padded to a fixed length (600 epochs per subject).

- **Merging:**
  - Sleep stage sequences and demographic features were joined using `sid6`.
  - Ensured data consistency and resolved missing keys.
  - Final input shapes:
    - `X_seq`: (118, 600)
    - `X_demo`: (118, 12)
    - `y`: (118,)

## Model Architecture

- **Input 1:** LSTM layer on `X_seq` (sleep stage sequence).
- **Input 2:** Fully-connected layers on `X_demo` (demographic).
- **Fusion Layer:** Concatenated both streams before final regression output.
- **Loss Function:** MSELoss
- **Metric:** Mean Absolute Error (MAE)


## Training Details

- **Train/Val/Test split:** 80/10/10 stratified
- **Early stopping** based on validation MAE
- **Training visualization** using MAE and Loss plots
- **Sanity checks:** Asserted no NaNs, verified input shapes


## Key Insights

- Created a **correlation heatmap** to visualize how demographics relate to sleep efficiency.
- Identified moderate positive/negative correlations:
  - Age and BMI were mildly correlated.
  - Race and smoking status showed observable influence in certain cases.


## Contributors

- **Nattakan Owatwansakul (S224205628)** – Model development, data wrangling, training loop, evaluation.
- Team: Part of the MOP Capstone “Chameleon” Health Behaviour Subgroup.
- Team members: Rayudu, Vinitha, Bhavithra, Alen, Ajay, Devin, Rishith


## Recommendations

- Include more subjects with diverse race/health profiles to improve model generalisation.
- Augment the sequence input with features like stage transitions, sleep fragmentation.
- Explore SHHS Visit 2 and Sleep-EDF datasets to expand coverage.
- Integrate model outputs into a public health dashboard or individual sleep profiling tool.




