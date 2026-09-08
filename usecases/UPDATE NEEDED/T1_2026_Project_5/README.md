# **Urban Water Demand Forecasting with Climate Signals**

A comparative study of machine learning and deep learning approaches for urban water demand forecasting, integrating climate signals with historical water consumption data.

---

## Team Members

| Name | Model |
|------|-------|
| Wimansa Rajakaruna | SARIMA-LSTM Hybrid Model |
| Keyur Jiteshbhai Pipaliya | Hybrid TCN-BiLSTM Model with Multi-Head Self-Attention |
| Anupa Dihan Hansaja | Multivariate LSTM Model |

---

## Project Overview

Due to the substantial growth of the population and climate variability, ccurate urban water demand forecasting is essential for water resource management. This project develops and compares three different forecasting approaches, each integrating climate variables with historical water consumption data to improve forecast accuracy. The models are evaluated against common performance metrics to identify the most effective approach for urban water demand forecasting. 

Two real-world datasets were used in this study. Daily urban water consumption data was sourced from the Municipal Artificial Intelligence Applications Lab GitHub Repository (https://github.com/aildnont/water-forecast), representing city wide water demand for the **City of London Ontario, Canada** from July 2019 to September 2020. Historical climate data was obtained from the Open-Meteo Historical Weather API, which provides ERA5 reanalysis data produced by the European Centre for Medium-range Weather Forecasts (https://open-meteo.com). 

**Climate Variables:**

| Variable | Unit |
|----------|------|
| Mean Temperature | °C |
| ET₀ (Evapotranspiration) | mm |
| Solar Radiation | MJ/m² |
| Wind Speed | km/h |
| Rainfall | mm |
| Precipitation | mm |

---

## Models

---

### Model 1 — SARIMA-LSTM Hybrid Model
**Contributor: Wimansa Rajakaruna**

#### Approach

The hybrid model combines SARIMA statistical forecasting with LSTM deep learning to capture both linear seasonal patterns and nonlinear climate-driven demand variations.

```
SARIMA(2,0,1)(1,1,1,300)
       *       Captures linear trend and seasonal patterns
       *       Produces residuals (unexplained variation)

Residual Extraction

LSTM Model
        *       Learns from SARIMA residuals
        *       Uses 6 climate variables as inputs
        *       Predicts residual corrections

Final Prediction = SARIMA forecast + LSTM residual
```

#### LSTM Architecture

```
Input (lookback=30 days, 6 climate features)
        ↓
LSTM Layer 1 (64 units, return_sequences=True)
        ↓
Dropout (0.2)
        ↓
LSTM Layer 2 (32 units, return_sequences=False)
        ↓
Dropout (0.2)
        ↓
Dense Output Layer (1 unit)
Total Parameters: 30,625
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | MSE |
| Lookback Window | 30 days |
| Train/Val/Test Split | 70% / 10% / 20% |
| Early Stopping Patience | 10 epochs |
| Convergence Epoch | 23 |

#### SARIMA-LSTM Hybrid Results

| Metric | LSTM Test Period | Validation |
|--------|--------------------------------------|--------------------------|
| RMSE | 1,255 m³ | 16,236 m³ |
| MAE | 977 m³ | 12,241 m³ |
| R² | 0.9612* | −0.747 |

*R² = 0.9612 evaluated within SARIMA training range. Honest unseen evaluation gives R² = −0.747.

#### Feature Importance

| Rank | Feature |
|------|---------|
| 1 | Wind Speed |
| 2 | Rainfall |
| 3 | Solar Radiation |

#### Key Findings

- The standalone SARIMA model achieved an AIC of 5,273 but produced R² = −0.699, confirming that statistical approaches alone are insufficient for capturing nonlinear climate-driven demand patterns
- Wind speed was the most influential feature for residual prediction, distinct from EDA which identified temperature as the strongest demand predictor, reflecting the difference between predicting demand versus predicting SARIMA residuals
- Both SARIMA and hybrid models struggled to generalise to the unprecedented summer peak of 135,000 m³ in July 2020, highlighting the importance of complete annual cycle coverage in training data

#### Limitations

1. SARIMA configured with m=300 due to computational constraints. The theoretically optimal m=365 caused kernel memory overflow
2. LSTM trained on April 2018 to January 2020 only, with no summer peak exposure, limiting generalisation to extreme summer demand
3. R² = 0.9612 is partially optimistic as it falls within the SARIMA training range. True unseen evaluation gives R² = −0.747
4. More than 60-day gap (March to June 2017) resulted in exclusion of pre-gap data
6. Approximately 3 years of post-gap data may not capture all seasonal variations

#### Future Work

- Fit SARIMA with m=365 using higher memory computing resources
- Incorporate complete annual cycles including summer peak in LSTM training data
- Source longer historical datasets covering 10+ years of continuous records
- Extend study to multiple cities across different climate zones

---

### Model 2 — Hybrid TCN-BiLSTM Model with Multi-Head Self-Attention
**Contributor: Keyur Jiteshbhai Pipaliya**

#### Approach
A hybrid deep learning model combines Temporal Convolutional Network (TCN), which handles seasonal and periodic patterns, Bidirectional LSTM (BiLSTM), which captures sequential dependencies, and Multi-Head Self-Attention, which focuses on most relevant timesteps. 

```
Input (lookback= 45, 28 features)
        ↓
Convolutional layer
        ↓
TCN Residual Blocks x4 (dilations 1,2,4,8)
        ↓
Multi-Head Self-Attention (4 heads)
        ↓
BiLSTM (96 units)
        ↓
BiLSTM (48 units)
        ↓
Dense(64, Relu)
        ↓
Dropout (0.2)
        ↓
Dense(32, Relu)
        ↓
Dense(1)

```

#### Results

| Metric | Value |
|--------|-------|
| RMSE | 2742.12 |
| MAE | 1944.2941 |
| MAPE | 1.69% |
| R² | 0.9105 |

#### Key Findings

- The hybrid TCN-BiLSTM-Attention model achieved an R² of 0.9105 on completely unseen test data, explaining over 91% of the variance in daily water consumption
- MAPE of 2.15% confirms predictions were on average within approximately 2% of actual consumption values, considering strong performance for a real-world daily forecasting task
- Huber loss proved more effective than standard MSE as the training objective, providing robustness against outlier consumption spikes present in the dataset
- Leak-free autoregressive lag features combined with rolling consumption statistics (7-day, 14-day, 30-day means) were the most impactful feature group, contributing more to accuracy than climate features alone

#### Limitations

- The dataset contains approximately 800 training samples, which fundamentally limits deep model learning capacity
- A systematic under-prediction bias was observed during extreme summer peak demand periods, which Huber loss alone could not fully resolve

#### Future Work

- Extend the date range of both datasets to increase training samples 
- Implement walk-forward cross-validation across multiple train/test splits for more statistically robust performance estimates
- Explore Transformer-based architectures such as Informer or Temporal Fusion Transformer, which are purpose-built for long-sequence time series forecasting

---

### Model 3 — Multivariate LSTM
**Contributor: Anupa Dihan Hansaja**

#### Approach

The Multivariate LSTM model directly predicts daily water demand by feeding both historical consumption values and climate variables simultaneously into the LSTM input sequence. The model treats water demand forecasting as a multivariate time series problem, where past values of all seven variables collectively inform the next day's demand prediction. 

#### LSTM Architecture

```
LSTM Baseline Model

Input (lookback=30 days, 6 climate features)
        ↓
LSTM Layer 1 (64 units, return_sequences=True)
        ↓
Dropout (0.2)
        ↓
LSTM Layer 2 (32 units, return_sequences=False)
        ↓
Dense Output Layer (1 unit)

LSTM Tuned Model

Input (lookback=60 days, 6 climate features)
        ↓
LSTM Layer 1 (128 units, return_sequences=True)
        ↓
Dropout (0.3)
        ↓
LSTM Layer 2 (64 units, return_sequences=False)
        ↓
Dense Output Layer (1 unit)
```

#### Results

| Metric | Value |
|--------|-------|
| RMSE | 689.13 |
| MAE | 453.33 |
| R² | 0.9959 |

#### Key Findings

- The tuned Multivariate LSTM achieved the strongest performance with RMSE of 689.13 m³, MAE of 453.33 m³, and R² of 0.9959, explaining approximately 99.6% of variance in daily water demand
- Incorporating climate variables consistently improved forecast accuracy over the univariate demand, only baseline across all evaluation metrics, confirming that climate signals carry meaningful predictive information
- Increasing the lookback window from 30 to 60 days and expanding LSTM units from 64 to 128 produced meaningful improvements in both RMSE and MAE
- ET₀ and solar radiation were identified as the climate variables most strongly correlated with prediction residuals, suggesting these variables carry the most unexplained predictive signal

#### Limitations

- Only the seven available raw climate variables were used without additional feature engineering such as lag features, rolling statistics, or cyclical time encodings
- Evaluation based on a single 80/20 chronological split may not fully capture performance variability across different time periods or seasonal conditions

#### Future Work

- Incorporate feature engineering techniques including autoregressive lag features, rolling consumption statistics, and cyclical time encodings to further improve model performance
- Explore attention mechanisms to improve capture of long-range seasonal dependencies in the demand series

---

## Model Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| SARIMA-LSTM (within training range) | 1,255 m³ | 977 m³ | 0.9612* |
| TCN-BiLSTM- Attention | 3,627 m³ | 2,465 m³ | 0.8433 |
| Multivariate LSTM (tuned) | 689 m³ | 453 m³ | 0.9959 |

*Note: SARIMA-LSTM R² = 0.9612 evaluated within SARIMA training range — not fully independent evaluation.

A direct comparison of all three modelling approaches reveals that the Multivariate LSTM achieved the strongest overall performance, with the tuned model producing an $R^2$ of 0.9959. 

However, a fully fair comparison is constrained by differences in test periods and training data across the three models. The Multivariate LSTM and TCN-BiLstm models were trained and evaluated on the full dataset spanning 2015 to 2020, which includes complete annual cycles and multiple summer peaks. But, the SARIMA-LSTM hybrid model was trained on the post-gap subset from 2017 to 2020 and evaluated on an autumn, winter period only. 

These findings suggest that end-to-end deep learning approaches such as the Multivariate LSTM and TCN-BiLSTM, which directly model demand from climate variables without a statistical preprocessing phase, can be more robust for forecasting across complete annual cycles when sufficient training data is available. The SARIMA-LSTM hybrid model demonstrated strong performance within its trained seasonal range but requires complete annual cycle converge in the training data to achieve comparable generalisation. 

---

## Requirements

```bash
# Core libraries
pip install pandas numpy matplotlib seaborn
pip install statsmodels scikit-learn
pip install jupyter ipykernel

# TensorFlow — Apple Silicon
pip install tensorflow-macos tensorflow-metal

# TensorFlow — Windows/Linux
pip install tensorflow
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Chameleon-company/MOP-Code.git
cd MOP-Code/Playground/Project_5

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows

# 3. Add datasets
# → water_consumption_data.csv
# → london_climate.csv

# 4. Run Individual Notebooks
jupyter notebook Urban_Water_Demand_Prediction_with_Climate_Signals.ipynb
jupyter notebook Urban Water Demand Prediction with Climate Signals_Multivariate LSTM.ipynb
jupyter notebook tcn_bilstm_forecasting.ipynb
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn, gridspec |
| Statistical Modelling | statsmodels (SARIMAX) |
| Machine Learning | scikit-learn |
| Scaling | MinMaxScaler, RobustScaler |
| Deep Learning | TensorFlow, Keras |
| LSTM | Keras Sequential, Bidirectional LSTM |
| TCN | Causal dilated Conv1D, residual blocks |
| Attention | Multi-Head Self-Attention |
| Callbacks | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Environment | VSCode, Google Colab |

---

## References

- Open-Meteo Historical Weather API — https://open-meteo.com
- Municipal AI Applications Lab — https://github.com/aildnont/water-forecast
