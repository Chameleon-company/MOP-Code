# Melbourne Traffic Congestion Forecasting Assistant

**Author:** Qizhe Yew  
**Student ID:** s222441776
**Level:** Intermediate  
**Pre-requisite skills:** Python

## Scenario

In a smart city context, commuters and transport planners need timely insight into when and where congestion is likely to occur. This project combines Melbourne Transport Activity Count data with temporal patterns, historical microclimate data, and optional live Open-Meteo weather values to predict whether a selected sensor location is likely to experience **Low**, **Medium**, or **High** congestion.

If the live weather API is unavailable, the system falls back to historical median weather values so predictions can still run. The system also adds an LLM and RAG layer so predictions are not only produced by a machine learning model, but also explained in plain English with practical travel or planning advice.

## Project Pipeline

### 1. Preprocessing

The notebook loads Transport Activity Count ZIP files and Melbourne microclimate sensor data, checks the raw datasets, cleans timestamps, removes duplicates, filters vehicle classes, and aggregates 5-minute traffic records into hourly counts.

Congestion labels are created per location, meaning a **High** label represents unusually high traffic for that specific sensor rather than a single city-wide threshold.

### 2. Model Training

The project trains and compares several classification models:

- Logistic Regression as a baseline
- Random Forest for nonlinear feature relationships
- XGBoost as the final stronger tabular model

The data is split by time so future records are not leaked into training.

### 3. Fine Tuning

XGBoost is fine-tuned using hyperparameter search. The final model is retrained on the training and validation periods, then evaluated on the held-out test period.

### 4. LLM Explanation

The trained model predicts a congestion class and probabilities for each class. For prediction-time context, the system uses live/forecast Open-Meteo weather values, with historical median weather as a fallback. Gemini is used first for natural-language explanations, with Groq as a fallback if Gemini is unavailable or rate limited.

### 5. Prompt Engineering

The LLM prompt includes the selected location, prediction time, class probabilities, time features, location traffic history, weather values, weather source, and model confidence so the explanation stays grounded in the model output.

### 6. RAG

A small traffic knowledge base is used with semantic search to retrieve relevant context. The retrieved context is passed into the LLM so the final response can be compared against a normal non-RAG response.

## Files

- `UC00218_Melbourne Traffic Congestion Forecasting Assistant.ipynb` - main notebook pipeline
- `traffic_map_ui.py` - Streamlit map interface
- `dependencies/` - downloaded (in case where API fails) and processed datasets
- `congestion_model.pkl`, `scaler.pkl`, `loc_map.pkl` - saved model artifacts
- `variable.env` - local API keys for Gemini/Groq

## Requirements

Install the main Python dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests joblib python-dotenv google-genai groq sentence-transformers
```

For the map UI:

```bash
pip install streamlit folium streamlit-folium
```

Create a `variable.env` file with:

```bash
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

## How to Run

Run the notebook from top to bottom:

```text
UC00218_Melbourne Traffic Congestion Forecasting Assistant.ipynb
```

To run the map UI after the model artifacts and dataset exist:

```bash
streamlit run traffic_map_ui.py
```

## Conclusion

This project demonstrates a city traffic prediction workflow: dataset preparation, location-specific congestion labelling, model training, model evaluation, live/fallback weather context, LLM explanation, and RAG-enhanced response generation.

## Areas for Improvement

- Add more years of transport data to improve generalisation.
- Include events, roadworks, public holiday, or incident data.
- Improve live data support if a reliable real-time transport API can be implemnented.
- Tune congestion label thresholds and compare against regression-based traffic volume prediction.
- Further improve the UI with clearer sensor grouping and user-friendly confidence explanations. And add the LLM explanation feature.
