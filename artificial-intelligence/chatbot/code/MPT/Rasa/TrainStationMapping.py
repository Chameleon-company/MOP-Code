''' 
	Author: AlexT
	Create full Melbourne Train Stations Map	
'''
import pandas as pd

# Define the dataset path
dataset_path = '/content/drive/MyDrive/Colab Notebooks/764/mpt_ds'

# Load GTFS data into DataFrames
stops_df = pd.read_csv(f'{dataset_path}/stops.txt')
# Extract relevant columns for the map
stops_map_df = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

#Import library for the Train Station Mapping
import folium

# Create a map centered around Melbourne
melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

# Add markers for each train station
for _, row in stops_map_df.iterrows():
    folium.Marker(
        location=[row['stop_lat'], row['stop_lon']],
        popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
        tooltip=row['stop_name']
    ).add_to(melbourne_map)

# Save the map to an HTML file
melbourne_map.save('melbourne_train_stations_map.html')
melbourne_map