import os
import pandas as pd
import xgboost as xgb

# Defining the paths for the models and data
BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIRECTORY, "notebooks", "xgb_wpfp.json")
DATA_PATH = os.path.join(BASE_DIRECTORY, "data", "processed", "kitchener_model_ready.csv")

# Categorical columns to convert to category dtype for XGBoost
cat_cols = [
    'pressure_zone', 'category', 'material', 'lined_material',
    'acquisition', 'ownership', 'rel_cleaning_subarea'
]

# Load the trained XGBoost model
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

# Preprocess input dataframe to handle categorical columns
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Removing the target column if it is present
    if "has_break" in df.columns:
        df = df.drop(columns=["has_break"])

    # Converting the categorical columns to category dtype
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

# predicting failure probabilities with risk levels for the input dataframe
def predict_failure(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    df_processed = preprocess(df)

    # Predicting probabilities
    probability = model.predict_proba(df_processed)[:, 1]

    # Adding the results to the dataframe
    df["failure_probability"] = probability
    df["risk_level"] = pd.cut(
        probability,
         # Defining the thresholds for Low, Medium and High risk
        bins=[0, 0.3, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    return df
# Loading data and predicting failure probabilities and risk levels
pipes = pd.read_csv(DATA_PATH)
pipes = predict_failure(pipes)
pipes["pipe_id"] = pipes.index.astype(str)

