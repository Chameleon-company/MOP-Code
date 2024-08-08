#!/usr/bin/env python
# coding: utf-8

# In[11]:


import requests
import folium
from geopy.geocoders import Nominatim
from IPython.display import display
import openrouteservice
from scipy.spatial import KDTree
from geopy.distance import geodesic
import pandas as pd
import webbrowser


def geocode_address(address):
    geolocator = Nominatim(user_agent="your_app_name")  #update
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        print("Address not found")
        return None


def visualize_combined_route(start_walking, transport, end_walking):
    start_location = start_walking['features'][0]['geometry']['coordinates'][0]
    m = folium.Map(location=[start_location[1], start_location[0]], zoom_start=13)
    
    folium.GeoJson(start_walking, name="start walking route", style_function=lambda x: {'color': 'green'}).add_to(m)
    folium.GeoJson(transport, name="public transport route", style_function=lambda x: {'color': 'blue'}).add_to(m)
    folium.GeoJson(end_walking, name="end walking route", style_function=lambda x: {'color': 'red'}).add_to(m)
    
    folium.LayerControl().add_to(m)
    map_file = 'map.html'
    m.save(map_file)
    
    # Open the map in the default web browser
    webbrowser.open_new_tab(map_file)
    print(f"Map has been saved and opened: {map_file}")


def get_route(start_lat, start_lng, end_lat, end_lng, api_key, profile='driving-car'):
    client = openrouteservice.Client(key=api_key)
    coordinates = [[start_lng, start_lat], [end_lng, end_lat]]
    route = client.directions(coordinates=coordinates, profile=profile, format='geojson')
    return route


def find_nearest_station(lat, lon, kdtree, df):
    distance, index = kdtree.query([lat, lon])
    nearest_station = df.iloc[index]
    nearest_station_coords = (nearest_station["stop_lat"], nearest_station["stop_lon"])
    point_coords = (lat, lon)
    distance_meters = geodesic(point_coords, nearest_station_coords).meters
    return nearest_station, distance_meters, distance

def main():
    # Loading test dataset
    #df = pd.read_csv('stops.txt')
    stops_file_path = r'C:\Users\logan\Desktop\Uni\Team proj\base model with map\actions\stops.txt'

    # Read the stops.txt file
    df = pd.read_csv(stops_file_path)

    # Extract the coordinates from the DataFrame
    coords = df[["stop_lat", "stop_lon"]].values

    # Create the k-d tree
    kdtree = KDTree(coords)

    user_input = input("Please enter your current location: ") #"fitzroy victoria"
    location = geocode_address(user_input)
    if location:
        print(f"Geocoded location: Latitude = {location[0]}, Longitude = {location[1]}")
        # visualize_location(location[0], location[1])

        # Find nearest station
        nearest_station, distance_meters, distance = find_nearest_station(location[0], location[1], kdtree, df)

        

        destination_input = input("Please enter your destination: ") #"fitzroy victoria" #"collingwood victoria"
        destination = geocode_address(destination_input)

        start_transport_lat = nearest_station["stop_lat"]
        start_transport_long = nearest_station["stop_lon"]

        if destination:
            #print(f"Geocoded destination: Latitude = {destination[0]}, Longitude = {destination[1]}")

            print(f"The nearest public transport stop is at {nearest_station['stop_name']} it is {distance_meters} meters away")

            nearest_station_destination, distance_meters_destination, distance_destination = find_nearest_station(destination[0], destination[1], kdtree, df)

            print(f"The nearest public transport stop to the destination is at {nearest_station_destination['stop_name']} it is {distance_meters_destination} meters away from the destination")

            print(f"Take public transport from {nearest_station['stop_name']} to {nearest_station_destination['stop_name']}")

            # Get the walking routes to and from public transport stops
            api_key = '5b3ce3597851110001cf6248a6b7c97bb850491794bb504b30e2f2f7'
            walking_route_start = get_route(location[0], location[1], start_transport_lat, start_transport_long, api_key, profile='foot-walking')
            walking_route_end = get_route(nearest_station_destination["stop_lat"], nearest_station_destination["stop_lon"], destination[0], destination[1], api_key, profile='foot-walking')

            # Get the public transport route
            transport_route = get_route(start_transport_lat, start_transport_long, nearest_station_destination["stop_lat"], nearest_station_destination["stop_lon"], api_key, profile='driving-car')

            # Visualize the combined routes
            visualize_combined_route(walking_route_start, transport_route, walking_route_end)
        else:
            print("Destination location could not be determined.")
    else:
        print("Current location could not be determined.")


if __name__ == "__main__":
    main()



