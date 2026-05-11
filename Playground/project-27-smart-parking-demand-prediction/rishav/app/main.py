from fastapi import FastAPI
import pandas as pd
import os

app = FastAPI()

sensors_df = None
events_df = None

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

@app.get("/")
def home():
    return {
        "message": "Melbourne Parking Data API",
        "sensors_rows": len(sensors_df) if sensors_df is not None else 0,
        "events_rows": len(events_df) if events_df is not None else 0
    }

@app.get("/sensors")
def get_sensors(limit: int = 1000):
    if sensors_df is None or len(sensors_df) == 0:
        return {"error": "No sensors data", "data": []}
    return {
        "total": len(sensors_df),
        "limit": min(limit, len(sensors_df)),
        "data": sensors_df.head(limit).to_dict(orient="records")
    }

@app.get("/events")
def get_events(limit: int = 1000):
    if events_df is None or len(events_df) == 0:
        return {"error": "No events data", "data": []}
    return {
        "total": len(events_df),
        "limit": min(limit, len(events_df)),
        "data": events_df.head(limit).to_dict(orient="records")
    }

@app.get("/columns")
def get_columns():
    return {
        "sensors_columns": list(sensors_df.columns) if sensors_df is not None else [],
        "events_columns": list(events_df.columns) if events_df is not None else []
    }
