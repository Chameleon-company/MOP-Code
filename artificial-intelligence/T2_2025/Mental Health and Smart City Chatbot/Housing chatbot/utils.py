from typing import Any

from pandas import DataFrame
import json

from geopy.geocoders import Nominatim
from scipy.spatial import KDTree
from geopy.distance import geodesic
import openrouteservice
import requests
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
# import warnings
# warnings.filterwarnings("ignore")

gtfs_api_key = os.getenv("GTFS_API_KEY")

class HousingData:
    @staticmethod
    def preprocess_address(data: DataFrame) -> DataFrame:
        """Preprocess the format of address to work with geolocator."""
        
        # Standardise the suburb name
        data["suburb_formatted"] = data["Suburb"].str.replace(" ", "").str.lower()
        
        # Remove any part before "/" in property number
        data["address_formatted"] = data["Address"].str.split("/").str[-1]
        
        # Remove the postcode in the end of the address
        data["address_formatted"] = data["address_formatted"].apply(lambda x:  ",".join(x.split(",")[:-1]) + ", Vic")
        
        return data
    
    @staticmethod
    def get_coordination(data: DataFrame) -> DataFrame:
        """Convert the address into geographic coordinations."""

        if "latitude" in data.columns and "longitude" in data.columns:
            pass # The data is already processed, don't need to process
        else:
            # Get the lat, long of the property
            data['coords'] = data['address_formatted'].apply(API_GTFS.geocode_address)
            data = data.dropna(subset='coords')
            data[['latitude', 'longitude']] = pd.DataFrame(
                data['coords'].tolist(), index=data.index
            )
            data = data.drop(columns=['coords'])
        
        return data
    
    @staticmethod
    def preprocess_data(data: DataFrame, output_pth: str) -> DataFrame:
        """Format the addresses and convert them into coordinations."""
    
        data = HousingData.preprocess_address(data)
        data = HousingData.get_coordination(data)
        data.to_csv(output_pth)
    
    @staticmethod
    def filter_df(data: DataFrame, filter_col: str, operator: str, values: Any) -> DataFrame:
        """Filter the dataframe based on the values from a column."""
        
        if operator == "=":
            data = data[data[filter_col] == values]
        elif operator == ">":
            data = data[data[filter_col] > values]
        elif operator == ">=":
            data = data[data[filter_col] >= values]
        elif operator == "<":
            data = data[data[filter_col] < values]
        elif operator == "<=":
            data = data[data[filter_col] <= values]
        elif operator == "in":
            data = data[data[filter_col].isin(values)]
            
        return data
    
    @staticmethod
    def filter_basic_data(data: DataFrame, json_data: json) -> DataFrame:
        """Filter the listing based on the entities from a json file."""
        
        # Area
        area_formatted = json_data["area"].replace(" ", "").lower()
        data = HousingData.filter_df(data, "suburb_formatted", "=", area_formatted)
        
        # Rental fee
        data = HousingData.filter_df(data, "Price pw", ">=", json_data["min_rental_fee_per_week"])
        data = HousingData.filter_df(data, "Price pw", "<=", json_data["max_rental_fee_per_week"])
        
        # Num bedrooms
        data = HousingData.filter_df(data, "Number of bedrooms", ">=", json_data["min_num_bedrooms"])
        data = HousingData.filter_df(data, "Number of bedrooms", "<=", json_data["max_num_bedrooms"])
        
        # Num bathrooms
        data = HousingData.filter_df(data, "Number of bathrooms", ">=", json_data["min_num_bathrooms"])
        
        # Num carspaces
        data = HousingData.filter_df(data, "Number of carspaces", ">=", json_data["min_num_carspaces"])
        
        # Housing types
        data = HousingData.filter_df(data, "Type of property", "in", json_data["property_type"])
        
        return data
    
    @staticmethod
    def filter_distance(data: DataFrame, json_data: json) -> DataFrame:
        """Filter the listing based on the distance to entities from a json file."""
        
        distance_data = json_data["close_to"]
        if len(distance_data) == 0:
            pass
        else:
            pass
        
        return data
    
    @staticmethod
    def rank_properties(data: DataFrame) -> DataFrame:
        """Calculate the score and rank the houses."""
        
        data["score"] = -0.001 * data["Price pw"] \
                        + 0.5  * data["Number of bedrooms"] \
                        + 0.25 * data["Number of bathrooms"] \
                        + 0.25 * data["Number of carspaces"]
                    
        return data
    
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
    def download_school_data(): 
        download_link = 'dv378_DataVic-SchoolLocations-2024'
        base_url = 'https://www.education.vic.gov.au/Documents/about/research/datavic/'
        dataset_id = dataset_id
        format = 'csv'

        url = f'{base_url}{download_link}.{format}'
        params = {
            'select': '*',
            'limit': -1,  # all records
            'lang': 'en',
            'timezone': 'UTC'
        }

        # GET request
        response = requests.get(url, params=params)

        if response.status_code == 200:
            # StringIO to read the CSV data
            url_content = response.text
            school_locations = pd.read_csv(StringIO(url_content), delimiter=',')
            school_locations= school_locations.dropna(subset=['Y', 'X'])
            print(school_locations.sample(10, random_state=999)) # Test
            return school_locations 
        else:
            return (print(f'Request failed with status code {response.status_code}'))

    @staticmethod
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

        return stop_times, train_stops, routes, trips


    @staticmethod
    def geocode_address(address):
        """Function to geocode an address using Nominatim."""
        
        geolocator = Nominatim(user_agent="mapping_app1.0", timeout=15)
        
        # Define the bounding box for Melbourne
        melbourne_bbox = [(-38.5267, 144.5937), (-37.5113, 145.5125)] 
        
        # Geocode the address within the Melbourne bounding box
        location = geolocator.geocode(address, viewbox=melbourne_bbox, bounded=True)
        
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
        
    @staticmethod
    def find_nearest_station(lat, lon, kdtree, df):
        """Query the KDTree with the given latitude and longitude to find nearest station. Distance is returned in degrees so need to calculate the meters."""
        
        distances, indices = kdtree.query([[lat, lon]], k=1)

        # Get the nearest station details from the DataFrame
        idx = int(indices[0]) 

        # Nearest station row
        nearest = df.iloc[idx]

        # Distance in meters
        distance_meters = geodesic(
            (lat, lon), (nearest["stop_lat"], nearest["stop_lon"])
        ).meters

        # Return ONLY stop_id + distance
        return nearest["stop_name"], distance_meters, nearest["stop_lat"], nearest["stop_lon"]
    
    @staticmethod    
    def find_nearest_school(lat, lon, kdtree, df):
        """Query the KDTree with the given latitude and longitude to find nearest station. Distance is returned in degrees so need to calculate the meters."""
        
        distance, index = kdtree.query([lat, lon])

        # Get the nearest station details from the DataFrame
        nearest_school = df.iloc[index]

        # Extract stations coords
        nearest_school_coords = (nearest_school["Y"], nearest_school["X"])
        point_coords = (lat, lon)

        # Calculate the geodesic distance (in meters) between the point and the nearest statio
        distance_meters = geodesic(point_coords, nearest_school_coords).meters
        school_name = nearest_school.get("School_Name", None)
        
        return school_name, nearest_school_coords[0], nearest_school_coords[1]
    
    def parse_time(gtfs_time: str) -> timedelta:

        hours, minutes, seconds = map(int, gtfs_time.split(':'))
        if hours >= 24:
            hours = hours - 24
            return timedelta(days=1, hours=hours, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

@staticmethod
def check_direct_route(stop_a, stop_b, stop_times_df):

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
def calculate_route_travel_time(route, stops_df, stop_times_df):

    total_travel_time = 0.0
    stop_times_df.sort_index(inplace=True)
    for i in range(len(route) - 1):
        station_a = route[i]
        station_b = route[i + 1]
        direct_route_exists, trip_ids = check_direct_route(station_a, station_b, stop_times_df)
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
def add_travel_time(has_direct_route,station_a,station_b, stops_df,stop_times_df):
    if has_direct_route:
        return calculate_route_travel_time([station_a]+[station_b], stops_df, stop_times_df)
    return calculate_route_travel_time([station_a] + ['Flinders Street Station'] + [station_b], stops_df, stop_times_df)

@staticmethod
def calculate_driving_time(start_lat, start_lng, end_lat, end_lng, api_key=gtfs_api_key, profile='driving-car'):
    # Create an OpenRouteService client instance
    client = openrouteservice.Client(key=api_key)

    # Set up coordinates for the route
    coordinates = [[start_lng, start_lat], [end_lng, end_lat]]
    # Get the route between the coordinates with the specified profile (e.g. can have walking/driving)
    route = client.directions(coordinates=coordinates, profile=profile, format='geojson')
    steps = route['features'][0]['properties']['segments'][0]['steps']
    total_distance = 0
    total_duration = 0

    # Generating step information
    for step in steps:
        instruction = step['instruction']
        distance = step['distance']
        duration = step['duration']

        # Accumulate total distance and duration
        total_distance += distance
        total_duration += duration
    
    total_duration = total_duration / 60
    return total_duration