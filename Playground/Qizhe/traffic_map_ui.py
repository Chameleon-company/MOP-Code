from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import re

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


st.set_page_config(page_title="Melbourne Traffic Map", layout="wide")

MELBOURNE_TZ = "Australia/Melbourne"
DATASET_PATHS = [Path("dependencies/final_dataset_melbourne.csv"), Path("final_dataset_melbourne.csv")]
FEATURES = [
    "location_encoded", "loc_mean_count", "loc_median_count", "loc_std_count", "loc_max_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_morning_peak", "is_evening_peak",
    "AirTemperature", "AverageWindSpeed", "RelativeHumidity", "PM25", "Noise",
]
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def dataset_path():
    for path in DATASET_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Run the notebook first so final_dataset_melbourne.csv exists.")


def make_melbourne_timestamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(MELBOURNE_TZ)
    return ts.tz_localize(MELBOURNE_TZ, ambiguous=False, nonexistent="shift_forward")


def parse_time(series):
    text = series.astype(str).str.strip()
    has_offset = text.str.contains(r"(Z|[+-]\d{2}:?\d{2})$", regex=True, na=False).mean() > 0.5
    if has_offset:
        return pd.to_datetime(text, errors="coerce", utc=True).dt.tz_convert(MELBOURNE_TZ)
    return pd.to_datetime(text, errors="coerce").dt.tz_localize(
        MELBOURNE_TZ, ambiguous=False, nonexistent="shift_forward"
    )


@st.cache_resource
def load_artifacts():
    return (
        joblib.load("congestion_model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("loc_map.pkl"),
    )


@st.cache_data
def load_data():
    df = pd.read_csv(dataset_path())
    df["time"] = parse_time(df["time"])
    for col in ["AirTemperature", "AverageWindSpeed", "RelativeHumidity", "PM25", "Noise"]:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
    df[["AirTemperature", "AverageWindSpeed", "RelativeHumidity", "PM25", "Noise"]] = (
        df[["AirTemperature", "AverageWindSpeed", "RelativeHumidity", "PM25", "Noise"]]
        .ffill().bfill().fillna(0)
    )
    return df.dropna(subset=["time"])


@st.cache_data
def build_locations(df):
    def clean(name):
        return re.sub(r"\s*[-–]\s*(?:Asset ID|CoM|COM|I-Hub)\s*[\w]+", "", str(name)).strip()

    locs = df.groupby("countlineName").agg(
        lat=("CountLocationLat", "first"),
        lon=("CountLocationLong", "first"),
        avg=("count", "mean"),
    ).reset_index()
    locs["display"] = locs["countlineName"].apply(clean)
    return locs.dropna(subset=["lat", "lon"]).sort_values("avg", ascending=False).reset_index(drop=True)


def weather_for(ts):
    try:
        params = {
            "latitude": -37.8136,
            "longitude": 144.9631,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": MELBOURNE_TZ,
            "forecast_days": 7,
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
        with urlopen(url, timeout=8) as resp:
            hourly = json.loads(resp.read().decode("utf-8"))["hourly"]
        times = pd.to_datetime(hourly["time"])
        idx = int(abs(times - ts.tz_localize(None).round("h")).argmin())
        return {
            "AirTemperature": float(hourly["temperature_2m"][idx]),
            "AverageWindSpeed": float(hourly["wind_speed_10m"][idx]),
            "RelativeHumidity": float(hourly["relative_humidity_2m"][idx]),
        }
    except Exception:
        return None


def feature_vector(location_id, ts, df, loc_map):
    hour, dow = ts.hour, ts.dayofweek
    loc_counts = df.loc[df["countlineName"] == location_id, "count"]
    same_hour = df[df["hour"] == hour] if "hour" in df.columns else df
    weather = weather_for(ts) or {
        "AirTemperature": float(same_hour["AirTemperature"].median()),
        "AverageWindSpeed": float(same_hour["AverageWindSpeed"].median()),
        "RelativeHumidity": float(same_hour["RelativeHumidity"].median()),
    }
    values = {
        "location_encoded": loc_map.get(location_id, -1),
        "loc_mean_count": float(loc_counts.mean()),
        "loc_median_count": float(loc_counts.median()),
        "loc_std_count": float(loc_counts.std()) if not pd.isna(loc_counts.std()) else 0.0,
        "loc_max_count": float(loc_counts.max()),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": int(dow in [5, 6]),
        "is_morning_peak": int(hour in [7, 8, 9]),
        "is_evening_peak": int(hour in [16, 17, 18]),
        "AirTemperature": weather["AirTemperature"],
        "AverageWindSpeed": weather["AverageWindSpeed"],
        "RelativeHumidity": weather["RelativeHumidity"],
        "PM25": float(df["PM25"].median()),
        "Noise": float(df["Noise"].median()),
    }
    return np.array([[values[f] for f in FEATURES]]), values


def predict(location_id, ts, df, model, scaler, loc_map):
    x, features = feature_vector(location_id, ts, df, loc_map)
    probs = model.predict_proba(scaler.transform(x))[0]
    pred = int(np.argmax(probs))
    return {
        "label": LABEL_MAP[pred],
        "confidence": round(float(probs[pred]) * 100, 1),
        "probabilities": {LABEL_MAP[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)},
        "features": features,
    }


def draw_map(locations, selected_id):
    selected = locations.loc[locations["countlineName"] == selected_id].iloc[0]
    fmap = folium.Map([selected["lat"], selected["lon"]], zoom_start=15, tiles="CartoDB positron")

    grouped = locations.copy()
    grouped["lat_key"] = grouped["lat"].round(6)
    grouped["lon_key"] = grouped["lon"].round(6)

    for _, group in grouped.groupby(["lat_key", "lon_key"], sort=False):
        group = group.sort_values("avg", ascending=False)
        primary = group.iloc[0]
        if (group["countlineName"] == selected_id).any():
            primary = group[group["countlineName"] == selected_id].iloc[0]
        is_selected = primary["countlineName"] == selected_id
        popup = "<br><br>".join(
            f"<b>{row.display}</b><br>Average: {row.avg:.1f} veh/hr"
            for row in group.head(8).itertuples()
        )
        folium.CircleMarker(
            [primary["lat"], primary["lon"]],
            radius=9 if is_selected else 5,
            color="#1565c0" if is_selected else "#6b7280",
            fill=True,
            fill_color="#1565c0" if is_selected else "#6b7280",
            fill_opacity=0.85 if is_selected else 0.55,
            weight=3 if is_selected else 1,
            tooltip=f"{len(group)} sensors here" if len(group) > 1 else primary["display"],
            popup=folium.Popup(popup, max_width=320),
        ).add_to(fmap)

    folium.Marker([selected["lat"], selected["lon"]], tooltip=selected["display"]).add_to(fmap)
    folium.Circle(
        [selected["lat"], selected["lon"]],
        radius=350,
        color="#1565c0",
        weight=1,
        fill=False,
        opacity=0.25,
    ).add_to(fmap)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[])


def main():
    model, scaler, loc_map = load_artifacts()
    df = load_data()
    locations = build_locations(df)

    st.title("Melbourne Traffic Congestion Map")

    with st.sidebar:
        query = st.text_input("Search street/sensor")
        filtered = locations
        if query.strip():
            filtered = locations[locations["display"].str.lower().str.contains(query.lower().strip(), na=False)]
        if filtered.empty:
            st.warning("No matching sensors.")
            filtered = locations

        selected_display = st.selectbox("Sensor", filtered["display"].tolist())
        selected = filtered.loc[filtered["display"] == selected_display].iloc[0]

        st.divider()
        st.subheader("Prediction time")
        date_value = st.date_input("Date", value=pd.Timestamp.now(tz=MELBOURNE_TZ).date())
        hour_value = st.selectbox(
            "Hour",
            options=list(range(24)),
            index=pd.Timestamp.now(tz=MELBOURNE_TZ).hour,
            format_func=lambda hour: f"{hour:02d}:00",
        )

    ts = make_melbourne_timestamp(pd.Timestamp(date_value.year, date_value.month, date_value.day, hour_value))
    result = predict(selected["countlineName"], ts, df, model, scaler, loc_map)

    map_col, info_col = st.columns([1.45, 1], gap="large")
    with map_col:
        draw_map(locations, selected["countlineName"])

    with info_col:
        st.subheader(selected_display)
        st.caption(ts.strftime("%A %d %B %Y, %I:%M %p"))
        st.metric("Predicted congestion", result["label"], f"{result['confidence']}% confidence")

        st.write("Class probabilities")
        for label in ["Low", "Medium", "High"]:
            st.progress(result["probabilities"][label] / 100, text=f"{label}: {result['probabilities'][label]}%")

        st.divider()
        st.write("Next 3 hours")
        rows = []
        for ahead in [1, 2, 3]:
            future_ts = ts + pd.Timedelta(hours=ahead)
            future = predict(selected["countlineName"], future_ts, df, model, scaler, loc_map)
            rows.append({"Time": f"+{ahead} hr", "Congestion": future["label"], "Confidence": f"{future['confidence']}%"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
