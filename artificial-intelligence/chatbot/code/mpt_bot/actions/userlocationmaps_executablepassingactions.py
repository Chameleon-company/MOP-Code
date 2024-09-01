#!/usr/bin/env python
# coding: utf-8

# In[11]:

###CODE BY LOGAN GUILDING


import requests
import folium
from geopy.geocoders import Nominatim
from IPython.display import display
import openrouteservice
from scipy.spatial import KDTree
from geopy.distance import geodesic
import pandas as pd
import webbrowser
import sys

#Function to geocode an address using Nominatim
def geocode_address(address):
    geolocator = Nominatim(user_agent="mapping_app1.0")
    #Geocode the input address to get latitude and longitude of address
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        print("Address not found")
        return None

#Function to visualize the combined route (walking + public transport) with Folium
def visualize_combined_route(start_walking, transport, end_walking):
    #Extract the starting location coordinates from the walking route
    start_location = start_walking['features'][0]['geometry']['coordinates'][0]
    
    #Create a Folium map centered at the starting location
    m = folium.Map(location=[start_location[1], start_location[0]], zoom_start=13)
    
    #Add the starting walking route to the map in green
    folium.GeoJson(start_walking, name="start walking route", style_function=lambda x: {'color': 'green'}).add_to(m)
    #Add the public transport route to the map in blue
    folium.GeoJson(transport, name="public transport route", style_function=lambda x: {'color': 'blue'}).add_to(m)
    #Add the ending walking route to the map in red
    folium.GeoJson(end_walking, name="end walking route", style_function=lambda x: {'color': 'red'}).add_to(m)
    
    #Adds a layer control to the map to toggle routes on and off
    folium.LayerControl().add_to(m)
    
    #Save the map as a html fil and open the map in the default browser
    map_file = 'map.html'
    m.save(map_file)
    
    webbrowser.open_new_tab(map_file)
    print(f"Map has been saved and opened: {map_file}")

#Function to get a route between two points using OpenRouteService
def get_route(start_lat, start_lng, end_lat, end_lng, api_key, profile='driving-car'):
    #Create an OpenRouteService client instance
    client = openrouteservice.Client(key=api_key)
    #Set up coordinates for the route
    coordinates = [[start_lng, start_lat], [end_lng, end_lat]]
    #Get the route between the coordinates with the specified profile (e.g. can have walking/driving)
    route = client.directions(coordinates=coordinates, profile=profile, format='geojson')
    #Return the route in GeoJSON format
    return route

#Function to find the nearest public transport station using KDTree algo
def find_nearest_station(lat, lon, kdtree, df):
    #Query the KDTree with the given latitude and longitude to find nearest station. Distance is returned in degrees so need to calculate the meters
    distance, index = kdtree.query([lat, lon])
    #Get the nearest station details from the DataFrame
    nearest_station = df.iloc[index]
    #Extract stations coords
    nearest_station_coords = (nearest_station["stop_lat"], nearest_station["stop_lon"])
    point_coords = (lat, lon)
    #Calculate the geodesic distance (in meters) between the point and the nearest statio
    distance_meters = geodesic(point_coords, nearest_station_coords).meters
    
    ###TO DO
    ###ADD FUNCTIONALTIY OF DIRECTIONS
    
    return nearest_station, distance_meters, distance

#Main function to execute the program by RASA
def main(current_location, destination_input):
    #Loading test dataset
    #df = pd.read_csv('stops.txt') *If we are able to use the same file location
    stops_file_path = r'C:\Users\logan\Desktop\Uni\Team proj\basemodelintegratedwithmap\actions\stops.txt'

    #Read the stops.txt file and extract coords from the DF
    df = pd.read_csv(stops_file_path)
    coords = df[["stop_lat", "stop_lon"]].values

    # Create the k-d tree
    kdtree = KDTree(coords)
    
    #Prompt the user to input their current location
    #user_input = input("Please enter your current location: ") #"fitzroy victoria"
    user_input = current_location
    #Geocode the user's inputted location
    location = geocode_address(user_input)
    #Check if geocoding was successful
    if location:
        #print(f"Geocoded location: Latitude = {location[0]}, Longitude = {location[1]}") Removed as causing logging flag on github
        # visualize_location(location[0], location[1])

        #Find the nearest public transport station to the current location
        nearest_station, distance_meters, distance = find_nearest_station(location[0], location[1], kdtree, df)

        
        #Prompt the user to input their destination
        #destination_input = input("Please enter your destination: ") #"fitzroy victoria" #"collingwood victoria"
        destination = geocode_address(destination_input)
        
        #Get the latitude and longitude of the nearest public transport station
        start_transport_lat = nearest_station["stop_lat"]
        start_transport_long = nearest_station["stop_lon"]

        #Check if destination geocoding was successful
        if destination:
            #print(f"Geocoded destination: Latitude = {destination[0]}, Longitude = {destination[1]}")

            print(f"The nearest public transport stop is at {nearest_station['stop_name']} it is {distance_meters} meters away")
            
            #Find the nearest public transport station to the destination
            nearest_station_destination, distance_meters_destination, distance_destination = find_nearest_station(destination[0], destination[1], kdtree, df)

            print(f"The nearest public transport stop to the destination is at {nearest_station_destination['stop_name']} it is {distance_meters_destination} meters away from the destination")

            print(f"Take public transport from {nearest_station['stop_name']} to {nearest_station_destination['stop_name']}")

            #Get the walking routes to and from public transport stops using OpenRouteService
            api_key = '5b3ce3597851110001cf6248a6b7c97bb850491794bb504b30e2f2f7'
            walking_route_start = get_route(location[0], location[1], start_transport_lat, start_transport_long, api_key, profile='foot-walking')
            walking_route_end = get_route(nearest_station_destination["stop_lat"], nearest_station_destination["stop_lon"], destination[0], destination[1], api_key, profile='foot-walking')

            #Get the public transport route between the two stations
            transport_route = get_route(start_transport_lat, start_transport_long, nearest_station_destination["stop_lat"], nearest_station_destination["stop_lon"], api_key, profile='driving-car')

            #Visualize the combined routes on a map
            visualize_combined_route(walking_route_start, transport_route, walking_route_end)
        else:
            print("Destination location could not be determined.")
    else:
        print("Current location could not be determined.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python userlocationmaps_executablepassingactions.py <current_location> <destination>")
        sys.exit(1)
    
    current_location = sys.argv[1]
    destination_input = sys.argv[2]
    main(current_location, destination_input)


