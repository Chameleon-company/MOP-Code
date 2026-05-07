import xgboost as xgb
import pandas as pd
import os

# Defining the paths for the models and data
BASE_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIRECTORY, "dashboard", "xgb_wpfp.json")

def load_model():
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    return model

def predict_risk(df_prepared: pd.DataFrame):
    model = load_model()

    # features the model expects
    required_features = [
        'pressure_zone', 'category', 'pipe_size', 'material', 'lined', 
        'lined_material', 'acquisition', 'ownership', 'bridge_main', 
        'criticality', 'rel_cleaning_area', 'rel_cleaning_subarea', 
        'undersized', 'shallow_main', 'condition_score', 'oversized', 
        'cleaned', 'shape__length', 'pipe_age'
    ]

    # Adding missing columns with default values
    for column in required_features:
        if column not in df_prepared.columns:
            if column in ['pressure_zone', 'category', 'material', 'lined_material', 
                      'acquisition', 'ownership', 'rel_cleaning_subarea']:
                df_prepared[column] = 'UNKNOWN'  # categorical defaults
            else:
                df_prepared[column] = 0   # numeric defaults
    
    # Ensure all required features are present
    X = df_prepared[required_features].copy()

    # === FORCE ALL CATEGORICAL COLUMNS TO 'UNKNOWN' ===
    cat_cols = ['pressure_zone', 'category', 'material', 'lined_material', 
                'acquisition', 'ownership', 'rel_cleaning_subarea']
    
    for col in cat_cols:
        if col in X.columns:
            X[col] = 'UNKNOWN'  # Force all categorical columns to 'UNKNOWN' to avoid unseen categories during prediction

    # Convert to category
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    print("All categorical columns forced to 'UNKNOWN'")
    
    # Predict
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    probability = model.predict(dmatrix)
    
    # Add results
    df_result = df_prepared.copy()
    df_result['failure_probability'] = probability
    df_result['risk_level'] = pd.cut(probability, 
                                     bins=[0, 0.35, 0.70, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
    return df_result

if __name__ == "__main__":
    df = pd.read_csv("/Users/homedesk/Documents/S779/2026/SIT764/capstone/MOP-Code/Playground/Project_3B_T126/notebooks/dashboard/melbourne_prepared_for_kitchener.csv")
    print("Input shape:", df.shape)
    
    result = predict_risk(df)
    
    result.to_csv("/Users/homedesk/Documents/S779/2026/SIT764/capstone/MOP-Code/Playground/Project_3B_T126/notebooks/dashboard/melbourne_with_risk_predictions.csv", index=False)
    
    print("\nSUCCESS! Predictions saved.")
    print(result['risk_level'].value_counts())
    print("\nSample:")
    print(result[['ASSET_ID', 'MATERIAL', 'PIPE_AGE', 'failure_probability', 'risk_level']].head(5))