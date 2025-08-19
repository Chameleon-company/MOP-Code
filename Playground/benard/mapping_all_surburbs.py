# mapping_all_suburbs.py
# Builds an interactive Folium map for ALL Melbourne geographies in my dataset.
# Saves: Playground/benard/health_access_map_all.html

import os
import math
import pandas as pd
import folium

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, "population_forecasts_clean.csv")
OUT_HTML = os.path.join(HERE, "health_access_map_all.html")

SUBURB_CENTROIDS = {
    "Carlton":         (-37.8000, 144.9660),
    "Carlton North":   (-37.7890, 144.9720),
    "Docklands":       (-37.8166, 144.9420),
    "East Melbourne":  (-37.8130, 144.9830),
    "Flemington":      (-37.7880, 144.9270),
    "Kensington":      (-37.7940, 144.9270),
    "Melbourne":       (-37.8136, 144.9631),   # CBD
    "North Melbourne": (-37.8000, 144.9430),
    "Parkville":       (-37.7860, 144.9560),
    "Port Melbourne":  (-37.8390, 144.9420),
    "Southbank":       (-37.8240, 144.9640),
    "South Yarra":     (-37.8400, 144.9930),
    "West Melbourne":  (-37.8080, 144.9420),
}

def main(target_year: int | None = None, min_radius: int = 4, max_radius: int = 28):
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Missing input: {IN_CSV}\nRun data_cleaning.py first.")

    df = pd.read_csv(IN_CSV)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["geography", "year", "value"]).copy()
    df["geography"] = df["geography"].astype(str).str.strip()

    if target_year is None:
        target_year = int(df["year"].max())

    df_year = df[df["year"] == target_year].copy()
    if df_year.empty:
        raise ValueError(f"No rows for year {target_year}.")

    agg = df_year.groupby("geography")["value"].sum().reset_index().sort_values("value", ascending=False)

    vmin, vmax = agg["value"].min(), agg["value"].max()
    def scale_radius(v):
        if vmax == vmin:
            return (min_radius + max_radius) / 2
        s = (math.sqrt(v) - math.sqrt(vmin)) / (math.sqrt(vmax) - math.sqrt(vmin))
        return min_radius + s * (max_radius - min_radius)

    m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12, control_scale=True)

    missing = []
    for _, row in agg.iterrows():
        suburb = row["geography"]
        total  = float(row["value"])
        coords = SUBURB_CENTROIDS.get(suburb)
        if not coords:
            missing.append(suburb); continue
        lat, lon = coords
        popup = folium.Popup(html=f"<b>{suburb}</b><br/>Year: {target_year}<br/>Forecast total: {int(total):,}", max_width=300)
        folium.CircleMarker([lat, lon], radius=scale_radius(total), fill=True, fill_opacity=0.6,
                            opacity=0.8, popup=popup, tooltip=f"{suburb}: {int(total):,}").add_to(m)

    folium.map.Marker(
        [-37.845, 144.935],
        icon=folium.DivIcon(html=(
            f"<div style='font-size:12px;background:white;padding:8px;border:1px solid #ccc;'>"
            f"<b>Mapping Access to Health Services – Population (Year {target_year})</b><br>"
            f"Circle size ∝ forecast population (sum of 'value'). "
            f"Missing centroids for: {len(set(missing))} suburbs"
            f"</div>"
        ))
    ).add_to(m)

    m.save(OUT_HTML)
    print(f"Saved map → {OUT_HTML}")
    if missing:
        print("No centroid for:", ", ".join(sorted(set(missing)))[:500], "...")

if __name__ == "__main__":
    main()
