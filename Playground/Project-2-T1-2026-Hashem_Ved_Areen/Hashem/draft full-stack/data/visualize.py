"""
Visualization module for the Smart Street Lighting system.

Generates maps (folium) and charts (matplotlib) showing:
- Pedestrian sensor locations and activity levels
- Streetlight positions and lux levels
- Efficiency classification (underlit/overlit/efficient)
- Proposed lighting layouts
"""

import pandas as pd
import numpy as np

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


def create_melbourne_basemap(
    center: tuple = (-37.8136, 144.9631),
    zoom: int = 14,
) -> "folium.Map":
    """Create a folium base map centered on Melbourne."""
    if not HAS_FOLIUM:
        raise ImportError("folium is required for map visualization. pip install folium")
    return folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")


def map_streetlights_and_sensors(
    ped_df: pd.DataFrame,
    light_df: pd.DataFrame,
    max_lights: int = 2000,
    max_sensors: int = 200,
) -> "folium.Map":
    """
    Create a map showing streetlight positions and pedestrian sensors.

    Streetlights colored by lux level, sensors colored by traffic volume.
    """
    m = create_melbourne_basemap()

    # Plot streetlights (orange circles, sized by lux)
    light_sample = light_df.dropna(subset=["Latitude", "Longitude"]).head(max_lights)
    for _, row in light_sample.iterrows():
        lux = row.get("lux_level", 0) or 0
        color = "darkgreen" if lux > 30 else ("orange" if lux > 10 else "red")
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.4,
            popup=f"Lux: {lux:.1f}",
        ).add_to(m)

    # Plot pedestrian sensors (blue, sized by traffic)
    sensor_summary = ped_df.groupby("sensor_name").agg(
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
        avg_traffic=("pedestriancount", "mean"),
    ).dropna().head(max_sensors)

    for _, row in sensor_summary.iterrows():
        radius = min(max(row["avg_traffic"] / 100, 3), 15)
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            popup=f"Sensor: {row.name}<br>Avg: {row['avg_traffic']:.0f}/hr",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;
                background: white; padding: 10px; border-radius: 5px;
                border: 1px solid #ccc; font-size: 12px;">
        <b>Legend</b><br>
        <span style="color: blue;">&#9679;</span> Pedestrian sensor (size = traffic)<br>
        <span style="color: darkgreen;">&#9679;</span> Streetlight (lux > 30)<br>
        <span style="color: orange;">&#9679;</span> Streetlight (lux 10-30)<br>
        <span style="color: red;">&#9679;</span> Streetlight (lux < 10)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def map_efficiency(
    ped_df: pd.DataFrame,
    light_df: pd.DataFrame,
) -> "folium.Map":
    """
    Create a map showing lighting efficiency classification.
    Underlit areas in red, overlit in orange, efficient in green.
    """
    m = create_melbourne_basemap()

    if "efficiency" not in ped_df.columns:
        print("Warning: Run analyze_lighting_efficiency() first.")
        return m

    color_map = {
        "Efficient": "green",
        "Adequate": "blue",
        "Underlit": "red",
        "Overlit": "orange",
    }

    sensor_eff = ped_df.groupby("sensor_name").agg(
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
        avg_traffic=("pedestriancount", "mean"),
        efficiency=("efficiency", lambda x: x.mode().iloc[0] if not x.mode().empty else "Adequate"),
    ).dropna()

    for _, row in sensor_eff.iterrows():
        color = color_map.get(row["efficiency"], "gray")
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{row.name}<br>Traffic: {row['avg_traffic']:.0f}/hr<br>Status: {row['efficiency']}",
        ).add_to(m)

    return m


def map_pedestrian_heatmap(ped_df: pd.DataFrame) -> "folium.Map":
    """Create a heatmap of pedestrian activity intensity."""
    m = create_melbourne_basemap(zoom=13)

    heat_data = ped_df.dropna(subset=["Latitude", "Longitude", "pedestriancount"])
    heat_data = heat_data[heat_data["pedestriancount"] > 0]

    heat_list = heat_data.apply(
        lambda row: [row["Latitude"], row["Longitude"], row["pedestriancount"]],
        axis=1,
    ).tolist()

    HeatMap(heat_list, radius=15, blur=10, max_zoom=16).add_to(m)

    return m


def map_proposed_design(
    design_dict: dict,
    center_lat: float = -37.8125,
    center_lon: float = 144.9785,
    bearing_deg: float = 0,
) -> "folium.Map":
    """
    Visualize a proposed lighting design on a map.

    Places light markers along a pathway based on the design specs.
    """
    m = create_melbourne_basemap(center=(center_lat, center_lon), zoom=17)

    num_lights = design_dict.get("num_lights", 10)
    spacing = design_dict.get("spacing_m", 20)
    pathway_length = design_dict.get("pathway_length_m", 200)

    # Generate light positions along a line
    # Convert spacing to approximate degrees (at Melbourne latitude)
    m_per_deg_lat = 111000
    m_per_deg_lon = 82000

    bearing_rad = np.radians(bearing_deg)
    for i in range(num_lights):
        distance_m = i * spacing
        if distance_m > pathway_length:
            break
        dlat = (distance_m * np.cos(bearing_rad)) / m_per_deg_lat
        dlon = (distance_m * np.sin(bearing_rad)) / m_per_deg_lon
        lat = center_lat + dlat
        lon = center_lon + dlon

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="yellow",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.9,
            popup=f"Light {i+1}<br>{design_dict.get('led_wattage', '?')}W LED",
            weight=2,
        ).add_to(m)

    # Draw pathway line
    start = [center_lat, center_lon]
    end_dlat = (pathway_length * np.cos(bearing_rad)) / m_per_deg_lat
    end_dlon = (pathway_length * np.sin(bearing_rad)) / m_per_deg_lon
    end = [center_lat + end_dlat, center_lon + end_dlon]
    folium.PolyLine([start, end], color="blue", weight=3, opacity=0.7).add_to(m)

    # Add info popup at start
    info = (
        f"<b>Proposed Design</b><br>"
        f"Lights: {num_lights}<br>"
        f"Spacing: {spacing}m<br>"
        f"Category: {design_dict.get('p_category', '?')}<br>"
        f"Technology: {design_dict.get('led_wattage', '?')}W LED<br>"
        f"Annual cost: ${design_dict.get('annual_energy_cost_aud', '?')}"
    )
    folium.Marker(
        location=start,
        popup=info,
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    return m


if __name__ == "__main__":
    from load_melbourne_data import load_pedestrian_data, load_streetlight_data
    from spatial_analysis import match_sensors_to_lights, analyze_lighting_efficiency

    print("Loading data...")
    ped = load_pedestrian_data(limit=2000)
    lights = load_streetlight_data(limit=5000)

    print("Running spatial analysis...")
    ped = match_sensors_to_lights(ped, lights)
    ped = analyze_lighting_efficiency(ped)

    print("Generating maps...")
    m1 = map_streetlights_and_sensors(ped, lights)
    m1.save("../outputs/map_sensors_and_lights.html")
    print("  Saved: outputs/map_sensors_and_lights.html")

    m2 = map_efficiency(ped, lights)
    m2.save("../outputs/map_efficiency.html")
    print("  Saved: outputs/map_efficiency.html")

    m3 = map_pedestrian_heatmap(ped)
    m3.save("../outputs/map_pedestrian_heatmap.html")
    print("  Saved: outputs/map_pedestrian_heatmap.html")

    # Test proposed design map
    design_dict = {
        "num_lights": 14, "spacing_m": 16.0, "pathway_length_m": 200,
        "p_category": "P9", "led_wattage": 30, "annual_energy_cost_aud": 352.80,
    }
    m4 = map_proposed_design(design_dict, bearing_deg=45)
    m4.save("../outputs/map_proposed_design.html")
    print("  Saved: outputs/map_proposed_design.html")

    print("Done! Open the HTML files in a browser to view maps.")
