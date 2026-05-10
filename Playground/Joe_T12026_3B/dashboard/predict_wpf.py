import pandas as pd
import os

def load_pipes():
    # Load the rich pre-computed file you have

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "melbourne_risk_llm_ready.csv"
    )

    df = pd.read_csv(DATA_PATH)
    
    # Standardizing column names 
    df = df.rename(columns={
        'PIPE_AGE': 'pipe_age',
        'MATERIAL': 'material',
        'PIPE_LENGTH': 'shape__length',
        'ASSET_ID': 'ASSET_ID',      
        'MAIN_NAME': 'MAIN_NAME'
    })

    # Create pipe_id for Streamlit selection if it does not already exist
    if "pipe_id" not in df.columns:
        if "ASSET_ID" in df.columns:
            df["pipe_id"] = df["ASSET_ID"].astype(str)
        else:
            df["pipe_id"] = df.index.astype(str)
    
    # uppercase risk levels for consistency
    if 'RISK_LEVEL' in df.columns:
        df['RISK_LEVEL'] = df['RISK_LEVEL'].str.upper()
    
    print(f"Loaded {len(df)} pipes")
    print("Available columns:", df.columns.tolist())
    print("\nRisk distribution:")
    print(df['RISK_LEVEL'].value_counts())
    
    return df

# Global variable used by the webapp to access the pipe data
pipes = load_pipes()