from fastapi import FastAPI
import pandas as pd
import os
import math

app = FastAPI()

sensors_df = None
events_df = None

# ========== HELPER FUNCTION ==========
def clean_nan_from_dict(obj):
    """
    Recursively convert NaN, Infinity values to None (becomes null in JSON).
    This fixes the "Out of range float values are not JSON compliant" error.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: clean_nan_from_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_from_dict(item) for item in obj]
    return obj

# ========== STARTUP ==========
@app.on_event("startup")
def load_data():
    global sensors_df, events_df
    
    print("Loading data...")
    
    # Load sensors data
    sensors_path = os.path.join(os.path.dirname(__file__), "../data/sensors_2014.csv")
    if os.path.exists(sensors_path):
        sensors_df = pd.read_csv(sensors_path)
        print(f"Loaded sensors: {len(sensors_df)} rows")
        print(f"Sensors columns: {list(sensors_df.columns)[:5]}...")
    else:
        print(f"File not found: {sensors_path}")
        sensors_df = pd.DataFrame()
    
    # Load events data
    events_path = os.path.join(os.path.dirname(__file__), "../data/events_2014.csv")
    if os.path.exists(events_path):
        events_df = pd.read_csv(events_path)
        print(f"Loaded events: {len(events_df)} rows")
        print(f"Events columns: {list(events_df.columns)[:5]}...")
    else:
        print(f"File not found: {events_path}")
        events_df = pd.DataFrame()

# ========== ENDPOINTS ==========
@app.get("/")
def home():
    return {
        "message": "Melbourne Parking Data API",
        "sensors_rows": len(sensors_df) if sensors_df is not None else 0,
        "events_rows": len(events_df) if events_df is not None else 0
    }

@app.get("/sensors")
def get_sensors(limit: int = 200, offset: int = 0):
    if sensors_df is None or len(sensors_df) == 0:
        return {"error": "No sensors data", "data": []}
    
    start = offset
    end = offset + min(limit, len(sensors_df) - offset)
    data_slice = sensors_df.iloc[start:end]
    
    raw_data = data_slice.to_dict(orient="records")
    cleaned_data = clean_nan_from_dict(raw_data)
    
    return {
        "total": len(sensors_df),
        "limit": limit,
        "offset": offset,
        "data": cleaned_data
    }

@app.get("/events")
def get_events(limit: int = 500, offset: int = 0):
    if events_df is None or len(events_df) == 0:
        return {"error": "No events data", "data": []}
    
    start = offset
    end = offset + min(limit, len(events_df) - offset)
    data_slice = events_df.iloc[start:end]
    
    raw_data = data_slice.to_dict(orient="records")
    cleaned_data = clean_nan_from_dict(raw_data)
    
    return {
        "total": len(events_df),
        "limit": limit,
        "offset": offset,
        "data": cleaned_data
    }

@app.get("/columns")
def get_columns():
    return {
        "sensors_columns": list(sensors_df.columns) if sensors_df is not None else [],
        "events_columns": list(events_df.columns) if events_df is not None else []
    }