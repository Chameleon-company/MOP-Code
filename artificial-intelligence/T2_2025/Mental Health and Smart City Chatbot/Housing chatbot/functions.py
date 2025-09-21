from geopy.geocoders import Nominatim
from scipy.spatial import KDTree
from geopy.distance import geodesic
import openrouteservice
import pandas as pd
import os
from io import StringIO
import requests
import seaborn as sns
import folium
import matplotlib.pyplot as plt
import geopandas as gpd
import json
import zipfile
from io import BytesIO
import re
from datetime import  timedelta
import functools

import warnings
warnings.filterwarnings("ignore")




class API_GTFS:
    @staticmethod
    def download_and_extract_data() -> dict:
        inner_zip_paths = ['2/google_transit.zip']
        url = 'https://data.ptv.vic.gov.au/downloads/gtfs.zip'
        required_files = ['stops.txt', 'stop_times.txt', 'routes.txt', 'trips.txt', 'calendar.txt']
        datasets = {}
        # Download main zip
        response = requests.get(url)
        response.raise_for_status()

        # Open main zip in memory
        with zipfile.ZipFile(BytesIO(response.content)) as main_zip:
            for inner_zip_path in inner_zip_paths:
                if inner_zip_path not in main_zip.namelist():
                    continue

                subfolder_name = os.path.basename(os.path.dirname(inner_zip_path))
                datasets[subfolder_name] = {}

                with main_zip.open(inner_zip_path) as inner_zip_file:
                    with zipfile.ZipFile(BytesIO(inner_zip_file.read())) as inner_zip:
                        for file_name in required_files:
                            if file_name in inner_zip.namelist():
                                with inner_zip.open(file_name) as f:
                                    datasets[subfolder_name][file_name] = pd.read_csv(f)

        return datasets

    @staticmethod
    @functools.lru_cache(maxsize=1) 
    def process_train_data():
        
        datasets = API_GTFS.download_and_extract_data()
        
        train_stops=datasets["2"]["stops.txt"]
        train_stops['stop_name'] = train_stops['stop_name'].astype(str).str.strip()
        train_stops = train_stops[train_stops['stop_name'].str.contains('Station', case=False, na=False)].copy()
        
        stop_times=datasets["2"]["stop_times.txt"]
        
        for df in (stop_times,):
            df["trip_id"]  = df["trip_id"].astype(str)
            df["stop_id"]  = df["stop_id"].astype(str)
        
        routes = datasets["2"]["routes.txt"]

        trips=datasets["2"]["trips.txt"]
        for df in (trips,):
            df["trip_id"]  = df["trip_id"].astype(str)
            df["route_id"] = df["route_id"].astype(str)
        train_data = stop_times.merge(train_stops, on="stop_id", how="left").merge(trips, on="trip_id", how="left").merge(routes, on="route_id", how="left")
        train_data = train_data.drop(columns=['location_type','parent_station','wheelchair_boarding','level_id','platform_code','wheelchair_boarding','level_id','platform_code','stop_headsign','pickup_type','drop_off_type','agency_id','service_id','route_type','route_color','route_text_color','shape_id','trip_headsign','direction_id','block_id','wheelchair_accessible'])
        train_data= train_data.dropna(subset=['stop_lat', 'stop_lon'])
        train_data = train_data.set_index(["stop_name", "trip_id"]).sort_index()

        return train_data, train_stops


    #Function to geocode an address using Nominatim
    @staticmethod
    def geocode_address(address):
        geolocator = Nominatim(user_agent="mapping_app1.0",timeout=15)
        
        #Define the bounding box for Melbourne
        melbourne_viewbox = [144.5937, -38.5267, 145.5125, -37.5113]
        
        #Geocode the address within the Melbourne bounding box
        location = geolocator.geocode(address, viewbox=melbourne_viewbox, bounded=True)
        
        if location:
            return location.latitude, location.longitude
        else:
            return None
        
    @staticmethod
    def find_nearest_station(lat, lon):
    #Query the KDTree with the given latitude and longitude to find nearest station. Distance is returned in degrees so need to calculate the meters
        train_data = API_GTFS.process_train_data()[0].reset_index()
        unique_stops = train_data.drop_duplicates(subset=["stop_name"])[["stop_name","stop_lat","stop_lon"]]
        kdtree = KDTree(unique_stops[["stop_lat","stop_lon"]].to_numpy())
        indices = kdtree.query([[lat, lon]], k=1)[1]

        #Get the nearest station details from the DataFrame
        idx = int(indices[0]) 

        # Nearest station row
        nearest = train_data.iloc[idx]
        stop_name = nearest.name[0]

        # Distance in meters
        distance_meters = geodesic(
            (lat, lon), (nearest["stop_lat"], nearest["stop_lon"])
        ).meters

        return stop_name, distance_meters, nearest["stop_lat"], nearest["stop_lon"]
    
    @staticmethod      
    def parse_time(gtfs_time: str) -> timedelta:

        hours, minutes, seconds = map(int, gtfs_time.split(':'))
        if hours >= 24:
            hours = hours - 24
            return timedelta(days=1, hours=hours, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def check_direct_route(stop_a, stop_b):
        
        stop_times_df= API_GTFS.process_train_data()[0]
        try:
            stop_a_times = stop_times_df.xs(stop_a, level='stop_name')
            stop_b_times = stop_times_df.xs(stop_b, level='stop_name')
        except KeyError:
            return False, []
        merged = pd.merge(stop_a_times.reset_index(), stop_b_times.reset_index(), on='trip_id', suffixes=('_a', '_b'))
        valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']]
        if not valid_trips.empty:
            return True, valid_trips['trip_id'].unique()
        return False, []

    @staticmethod
    def calculate_route_travel_time(route):

        total_travel_time = 0.0
        stop_times_df = API_GTFS.process_train_data()[0]
        stop_times_df.sort_index(inplace=True)
        for i in range(len(route) - 1):
            station_a = route[i]
            station_b = route[i + 1]
            direct_route_exists, trip_ids = API_GTFS.check_direct_route(station_a, station_b)
            if not direct_route_exists:
                return None
            best_trip_id = trip_ids[0]
            try:
                stop_a_time = stop_times_df.loc[(station_a, best_trip_id),'departure_time']
                stop_b_time = stop_times_df.loc[(station_b, best_trip_id),'arrival_time']
                if isinstance(stop_a_time, pd.Series):
                    stop_a_time = stop_a_time.iloc[0]
                if isinstance(stop_b_time, pd.Series):
                    stop_b_time = stop_b_time.iloc[0]
            except KeyError:
                return None
            travel_time = API_GTFS.parse_time(stop_b_time) - API_GTFS.parse_time(stop_a_time)
            total_travel_time += travel_time.total_seconds()
        return total_travel_time / 60  # Return time in minutes
    
    @staticmethod
    def add_travel_time(station_a,station_b):
        has_direct, _= API_GTFS.check_direct_route(station_a,station_b)
        if has_direct:
            return API_GTFS.calculate_route_travel_time([station_a]+[station_b])
        return API_GTFS.calculate_route_travel_time([station_a, 'Flinders Street Station', station_b])

    @staticmethod
    def calculate_driving_time(start_lat, start_lng, end_lat, end_lng,api_key, profile='driving-car'):
        #Create an OpenRouteService client instance
        client = openrouteservice.Client(key=api_key)

        #Set up coordinates for the route
        coordinates = [[start_lng, start_lat], [end_lng, end_lat]]
        #Get the route between the coordinates with the specified profile (e.g. can have walking/driving)
        route = client.directions(coordinates=coordinates, profile=profile, format='geojson')
        steps = route['features'][0]['properties']['segments'][0]['steps']
        total_distance = 0
        total_duration = 0

        #Generating step information
        for step in steps:
            distance = step['distance']
            duration = step['duration']

            #Accumulate total distance and duration
            total_distance += distance
            total_duration += duration
        
        total_duration = total_duration/60
        return total_duration, total_distance
    
    @staticmethod
    def calculate_walking(start_lat, start_lng, end_lat, end_lng,api_key, profile='foot-walking'):
        #Create an OpenRouteService client instance
        client = openrouteservice.Client(key=api_key)

        #Set up coordinates for the route
        coordinates = [[start_lng, start_lat], [end_lng, end_lat]]
        #Get the route between the coordinates with the specified profile (e.g. can have walking/driving)
        route = client.directions(coordinates=coordinates, profile=profile, format='geojson')
        steps = route['features'][0]['properties']['segments'][0]['steps']
        total_distance = 0
        total_duration = 0

        #Generating step information
        for step in steps:
            distance = step['distance']
            duration = step['duration']

            #Accumulate total distance and duration
            total_distance += distance
            total_duration += duration
        
        total_duration = total_duration/60
        return total_duration, total_distance
    
    def travel_time_between(start_address: str, end_address: str, by: str):
        """
        Computes travel time between two addresses by 'car' | 'walk' | 'train'.

        Returns a dict with total_minutes and a breakdown.
        Requires OPENROUTESERVICE_API_KEY (or ORS_API_KEY) for car/walk/train (walking legs).
        """
        
        if by not in {"car", "walk", "train"}:
            raise ValueError("Parameter 'by' must be one of: 'car', 'walk', 'train'.")

        # Geocode addresses
        start_coords = API_GTFS.geocode_address(start_address)
        end_coords   = API_GTFS.geocode_address(end_address)
        if not start_coords or not end_coords:
            raise ValueError("Could not find address within Melbourne")

        start_lat, start_lon = start_coords
        end_lat, end_lon     = end_coords

        # ORS key (needed for car, walk, and the walking legs of 'train')
        api_key = "1ktSQErBv5y6ykTlW0LmDKQ6cPH5yF8V"

        # Car
        if by == "car":
            minutes, meters = API_GTFS.calculate_driving_time(start_lat, start_lon, end_lat, end_lon, api_key, profile="driving-car")
            if minutes is None:
                raise RuntimeError("Driving route not found.")
            return {
                "mode": "car",
                "total_minutes": minutes,
                "segments": [
                    {"type": "car", "minutes": minutes, "meters": meters,
                    "from": {"lat": start_lat, "lon": start_lon}, "to": {"lat": end_lat, "lon": end_lon}}
                ]
            }

        # Walk
        if by == "walk":
            minutes, meters = API_GTFS.calculate_walking(start_lat, start_lon, end_lat, end_lon, api_key, profile="foot-walking")
            if minutes is None:
                raise RuntimeError("Walking route not found.")
            return {
                "mode": "walk",
                "total_minutes": minutes,
                "segments": [
                    {"type": "walk", "minutes": minutes, "meters": meters,
                    "from": {"lat": start_lat, "lon": start_lon}, "to": {"lat": end_lat, "lon": end_lon}}
                ]
            }

        # Train (walk -> station A) + (train A->B) + (walk -> destination)
        # Find nearest stations using your class helper
        station_a_name, a_dist_m, a_lat, a_lon = API_GTFS.find_nearest_station(start_lat, start_lon)
        station_b_name, b_dist_m, b_lat, b_lon = API_GTFS.find_nearest_station(end_lat, end_lon)

        # Walk to station A
        walk1_min, walk1_m = API_GTFS.calculate_walking(start_lat, start_lon, a_lat, a_lon, api_key, profile="foot-walking")
        if walk1_min is None:
            raise RuntimeError("Walking route (start -> nearest station) not found.")

        # Train A -> B (direct or via Flinders per your helper)
        train_min = API_GTFS.add_travel_time(station_a_name, station_b_name)
        if train_min is None:
            raise RuntimeError(f"No usable train route between '{station_a_name}' and '{station_b_name}'.")

        # Walk from station B to destination
        walk2_min, walk2_m = API_GTFS.calculate_walking(b_lat, b_lon, end_lat, end_lon, api_key, profile="foot-walking")
        if walk2_min is None:
            raise RuntimeError("Walking route (nearest station -> destination) not found.")

        total = float(walk1_min) + float(train_min) + float(walk2_min)
        return {
            "mode": "train",
            "total_minutes": total,
            "stations": {"origin_station": station_a_name, "dest_station": station_b_name},
            "segments": [
                {"type": "walk", "minutes": walk1_min, "meters": walk1_m,
                "from": {"lat": start_lat, "lon": start_lon}, "to": {"lat": a_lat, "lon": a_lon}},
                {"type": "train", "minutes": train_min, "from_station": station_a_name, "to_station": station_b_name},
                {"type": "walk", "minutes": walk2_min, "meters": walk2_m,
                "from": {"lat": b_lat, "lon": b_lon}, "to": {"lat": end_lat, "lon": end_lon}},
            ]
        }