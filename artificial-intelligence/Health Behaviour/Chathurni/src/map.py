import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from pathlib import Path

# Load Data
data_path = Path("D:\Chathurni\src\data\processed\subjective_wellbeing_cleaned_with_coordinates.csv")
predictions_path = Path("D:/Chathurni/src/data/predictions/predictions.csv")

# Combine historical and predicted data
data = pd.concat([pd.read_csv(data_path), pd.read_csv(predictions_path)], ignore_index=True)

data['Latitude'] = data['Subcategory'].apply(lambda x: -37.8136 + hash(x) % 10 / 100)  # Replace with real lat
data['Longitude'] = data['Subcategory'].apply(lambda x: 144.9631 + hash(x) % 10 / 100)  # Replace with real lon

def create_map(subtopic="All", year="All"):
    """
    Generate a Folium map based on the selected subtopic and year.
    """
    # Start with the full dataset
    filtered_data = data.copy()

    # Filter by subtopic if specified
    if subtopic != "All":
        filtered_data = filtered_data[filtered_data['Subtopics'] == subtopic]

    # Filter by year if specified
    if year != "All":
        filtered_data = filtered_data[filtered_data['year'] == int(year)]

    # Initialize the map centered on Melbourne
    m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12, tiles="cartodbpositron")

    # Add Markers with Tooltips and Popups
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_data.iterrows():
        color = "green" if row['Percentage'] > 75 else "orange" if row['Percentage'] > 50 else "red"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(
                f"<b>Suburb:</b> {row['Subcategory']}<br>"
                f"<b>Year:</b> {row['year']}<br>"
                f"<b>Percentage:</b> {row['Percentage']}%",
                max_width=300,
            ),
            tooltip=f"{row['Subcategory']} ({row['Percentage']}%)",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(marker_cluster)

    # Add Heatmap 
    heat_data = [
        [row['Latitude'], row['Longitude'], row['Percentage']]
        for _, row in filtered_data.iterrows() if row['Percentage'] > 0
    ]
    if heat_data:
        HeatMap(heat_data, radius=15, blur=10).add_to(m)

    return m
