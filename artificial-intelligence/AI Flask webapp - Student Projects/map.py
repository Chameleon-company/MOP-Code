import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from pathlib import Path

# Load Data
data_path = Path(r"D:\MOP-Code-mtortely\MOP-Code-mtortely\artificial-intelligence\AI Flask webapp - Student Projects\subjective_wellbeing_cleaned_with_coordinates.csv")
predictions_path = Path(r"D:\MOP-Code-mtortely\MOP-Code-mtortely\artificial-intelligence\AI Flask webapp - Student Projects\predictions.csv")

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

    # Filter by subtopic
    if subtopic != "All":
        filtered_data = filtered_data[filtered_data['Subtopics'] == subtopic]

    # Filter by year
    if year != "All":
        filtered_data = filtered_data[filtered_data['year'] == int(year)]

    # Initialize the map centered on Melbourne
    m = folium.Map(location=[-37.8136, 144.9631], zoom_start=12, tiles="cartodbpositron")

    # Layered Map with Clustering
    marker_cluster = MarkerCluster(name="Marker Clusters").add_to(m)
    heatmap_layer = folium.FeatureGroup(name="Heatmap").add_to(m)

    # Add Markers
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
        HeatMap(heat_data, radius=15, blur=10, gradient={0.2: "blue", 0.5: "yellow", 0.8: "green"}).add_to(heatmap_layer)

    # Add Mini-Legend in Table Format
    legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: auto; 
                 background-color: white; z-index:9999; font-size:14px;
                 border:2px solid grey; padding: 10px;">
     <b>Legend:</b><br>
     <table style="width:100%; border-collapse: collapse; text-align: left;">
         <tr>
             <td style="background-color: red; width: 20px; height: 20px;"></td>
             <td>Low Score (<50%)</td>
         </tr>
         <tr>
             <td style="background-color: orange; width: 20px; height: 20px;"></td>
             <td>Medium Score (50-75%)</td>
         </tr>
         <tr>
             <td style="background-color: green; width: 20px; height: 20px;"></td>
             <td>High Score (>75%)</td>
         </tr>
     </table>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control
    folium.LayerControl().add_to(m)

    return m
