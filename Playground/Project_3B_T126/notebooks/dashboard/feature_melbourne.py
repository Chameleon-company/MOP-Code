import pandas as pd
import numpy as np

def prepare_for_kitchener_model(df_melb):
    df = df_melb.copy()
    
    # Map Melbourne columns → Kitchener model features
    df['pipe_size'] = pd.to_numeric(df.get('PIPE_WIDTH'), errors='coerce')
    df['material'] = df.get('MATERIAL', 'UNKNOWN').astype('category')
    df['pipe_age'] = pd.to_numeric(df.get('PIPE_AGE'), errors='coerce')
    df['shape__length'] = pd.to_numeric(df.get('PIPE_LENGTH'), errors='coerce')
    
    # Fill missing values
    df['pipe_size'] = df['pipe_size'].fillna(300)
    df['pipe_age'] = df['pipe_age'].fillna(50)
    df['shape__length'] = df['shape__length'].fillna(100)
    
    # Additional features the model likely expects
    df['pressure_zone'] = df.get('MAIN_NAME', 'UNKNOWN').astype('category')
    df['category'] = df.get('MAIN_CLASS', 'TREATED').astype('category')
    df['lined'] = 0
    df['criticality'] = 6
    df['condition_score'] = 8.5 - (df['pipe_age'] / 20).clip(upper=5)
    
    # Keep important columns for dashboard
    df['ASSET_ID'] = df.get('ASSET_ID', df.get('COMPKEY'))
    df['MAIN_NAME'] = df.get('MAIN_NAME')
    
    # Final features (adjust based on your model's actual features)
    features = ['pressure_zone', 'category', 'pipe_size', 'material', 
                'lined', 'pipe_age', 'shape__length', 'criticality', 'condition_score']
    
        # Add all missing columns the model needs
    extra_columns = ['lined_material', 'acquisition', 'ownership', 'bridge_main',
                  'rel_cleaning_area', 'rel_cleaning_subarea', 'undersized',
                  'shallow_main', 'oversized', 'cleaned']
    
    for column in extra_columns:
        if column not in df.columns:
            if column in ['lined_material', 'acquisition', 'ownership', 'rel_cleaning_subarea']:
                df[column] = 'UNKNOWN'
            else:
                df[column] = 0
    
    return df[features + ['ASSET_ID', 'MAIN_NAME', 'PIPE_AGE', 'MATERIAL', 'PIPE_LENGTH']]

# ====================== execution ======================
if __name__ == "__main__":
    df = pd.read_csv("/Users/homedesk/Documents/S779/2026/SIT764/capstone/MOP-Code/Playground/Project_3B_T126/data/processed/melbourne_cleaned_for_adaptation.csv")
    prepared = prepare_for_kitchener_model(df)
    prepared.to_csv("/Users/homedesk/Documents/S779/2026/SIT764/capstone/MOP-Code/Playground/Project_3B_T126/notebooks/dashboard/melbourne_prepared_for_kitchener.csv", index=False)
    print(f"Saved melbourne_prepared_for_kitchener.csv with {len(prepared)} rows")
    print(prepared.columns.tolist())