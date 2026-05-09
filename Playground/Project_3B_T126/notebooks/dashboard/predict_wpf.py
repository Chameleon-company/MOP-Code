import pandas as pd

def load_pipes():
    # Load the rich pre-computed file you have
    df = pd.read_csv(".../data/processed/melbourne_risk_llm_ready.csv")
    
    # Standardizing column names 
    df = df.rename(columns={
        'PIPE_AGE': 'pipe_age',
        'MATERIAL': 'material',
        'PIPE_LENGTH': 'shape__length',
        'ASSET_ID': 'ASSET_ID',      
        'MAIN_NAME': 'MAIN_NAME'
    })
    
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