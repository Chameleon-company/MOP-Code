'''-------------------------------------------------------------------------------------------------------'''
''' -------------------------------------------------------------------------------------------------------
	GTFSUtils Singleton common methods can be use in actions - This is for train station only
	    - download_and_extract_data: Download the whole dataset from ptv (Author: AlexT)	    
	    - load_mode_data: Load data for a specific mode - train, tram or bus etc. (Author: AlexT)
	    - load_combined_data: load and combine data for train, tram and bus (Author: AlexT)
	    - find_common_routes:  Find common routes between two stops (Author: AlexT)
	    - load_data: - Author: AlexT (Download and load data for Train only)
		- normalise_gtfs_data: (Author: AlexT)
		- find_station_name: (Author: AlexT)
		- convert_gtfs_time: (Author: AlexT)
		- parse_time: (Author: AlexT)
		- get_station_id: (Author: AlexT)
		- extract_stations_from_query: (Author: AlexT)
		- check_direct_route: (Author: AlexT)
		- calculate_route_travel_time: (Author: AlexT)
		- calculate_transfers: (Author: AlexT)
		- find_best_route_with_transfers: (Author: AlexT)
		- handle_error: handle error and logging: (Author: AlexT)
		- generate_signature: signature required for PTV API: (Author: AlexT)
        - fetch_disruptions: (Author: AlexT)
        - filter_active_disruptions: (Author: AlexT)
        - check_route_and_fetch_disruptions: (Author: AlexT)
        - extract_route_name: Applicable for Tram, Bus and Train: (Author: AlexT)
        - determine_user_route: Determine the route (bus or tram): (Author: AlexT)
        - determine_schedule: Determine the schedule for a specific route (bus or tram): (Author: AlexT)
	-------------------------------------------------------------------------------------------------------
'''
import spacy
import folium
import os
import math
from io import BytesIO
import zipfile
import requests
import pandas as pd
import logging
from typing import Any, Text, Dict, List, Optional
from fuzzywuzzy import process, fuzz
from datetime import datetime, timedelta
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Tuple
from collections import deque
import certifi
from sanic import Sanic
from sanic.response import text
import re
import hashlib
import hmac
import urllib.parse
from tabulate import tabulate
from transformers import pipeline

# User ID and API Key
user_id = "3003120"
api_key = "0efc8af6-e6c8-445e-a426-fcad4aed37f2"

# API Base URL
base_url = "https://timetableapi.ptv.vic.gov.au"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

class GTFSUtils:
    @staticmethod
    def normalise_gtfs_data(stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> None:
        """
           Author: AlexT
            Normalise the stop names and ensure the stop_times DataFrame is indexed correctly.
        """
        stops_df['stop_name'] = stops_df['stop_name'].astype(str).str.strip()
        stops_df['stop_id'] = stops_df['stop_id'].astype(str).str.strip()

        stops_df['normalized_stop_name'] = stops_df['stop_name'].str.lower()

        stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(str).str.strip()
        expected_columns = ['stop_id', 'trip_id', 'arrival_time', 'departure_time']

        if all(col in stop_times_df.columns for col in expected_columns):
            try:
                stop_times_df.set_index(['stop_id', 'trip_id'], inplace=True)
            except KeyError as e:
                logger.error(f"Error setting index: {e}")
        else:
            logger.error("Expected columns are not present in the DataFrame.")
            logger.error("Available columns: %s", stop_times_df.columns)


    @staticmethod
    def download_and_extract_data(url: str, dataset_path: str, inner_zip_paths: list) -> bool:
        """
        Author: AlexT
        Downloads and extracts GTFS data from a provided URL, extracting each inner zip to its respective subfolder.
        :param url: The URL to download the GTFS zip file from.
        :param dataset_path: The folder where the extracted data will be stored.
        :param inner_zip_paths: A list of paths to inner zip files within the main zip file.
        :return: True if extraction was successful, False otherwise.
        """
        #LoganG updating so GTFS files arent downloaded each time
        # Check if the data already exists
        all_data_exists = True
        for inner_zip_path in inner_zip_paths:
            subfolder_name = os.path.basename(os.path.dirname(inner_zip_path))  
            subfolder_path = os.path.join(dataset_path, subfolder_name)
            
            # Check for GTFS files in each subfolder
            required_files = ['stops.txt', 'stop_times.txt', 'routes.txt', 'trips.txt', 'calendar.txt']
            for file in required_files:
                file_path = os.path.join(subfolder_path, file)
                if not os.path.exists(file_path):
                    all_data_exists = False
                    break
            
            if not all_data_exists:
                break

        # If all required files exist, skip download
        if all_data_exists:
            logger.info("GTFS data already exists in all subfolders. Skipping download.")
            return True

        # If data doesn't exist, proceed with download and extraction
        os.makedirs(dataset_path, exist_ok=True)

        try:
            logger.info("Downloading the zip file...")
            response = requests.get(url, verify=certifi.where())
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return False

        try:
            logger.info("Opening the main zip file...")
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                for inner_zip_path in inner_zip_paths:
                    logger.info(f"Searching for '{inner_zip_path}' inside the main zip file...")
                    if inner_zip_path in z.namelist():
                        logger.info(f"Found '{inner_zip_path}'. Extracting...")

                        # Create a specific subfolder for this inner zip
                        subfolder_name = os.path.basename(os.path.dirname(inner_zip_path))  # e.g., '2', '3', '4'
                        subfolder_path = os.path.join(dataset_path, subfolder_name)
                        os.makedirs(subfolder_path, exist_ok=True)

                        # Extract the inner zip file to its corresponding subfolder
                        with z.open(inner_zip_path) as inner_zip_file:
                            with zipfile.ZipFile(BytesIO(inner_zip_file.read())) as inner_z:
                                logger.info(f"Extracting contents of '{inner_zip_path}' to '{subfolder_path}'...")
                                inner_z.extractall(subfolder_path)
                    else:
                        logger.warning(f"'{inner_zip_path}' not found in the main zip file. Skipping...")
            logger.info("Extraction of all specified inner zip files complete.")
            return True
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return False

    @staticmethod
    def load_mode_data(path: str, mode: str) -> Optional[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Author: AlexT
        Load GTFS data files for a specific transport mode (train, tram, bus).
        :param path: Path to the folder containing GTFS files.
        :param mode: The transport mode (e.g., 'train', 'tram', 'bus').
        :return: A tuple of DataFrames (stops, stop_times, routes, trips, calendar) or None if an error occurs.
        """
        try:
            logger.info(f"Loading GTFS data for mode: {mode} from path: {path}")

            # Load required files
            stops = pd.read_csv(f"{path}/stops.txt")
            stop_times = pd.read_csv(f"{path}/stop_times.txt")
            routes = pd.read_csv(f"{path}/routes.txt")
            trips = pd.read_csv(f"{path}/trips.txt")
            calendar = pd.read_csv(f"{path}/calendar.txt")

            # # Add mode column
            # stops['mode'] = mode
            # stop_times['mode'] = mode
            # routes['mode'] = mode
            # trips['mode'] = mode

            # Validate and normalize the data
            GTFSUtils.normalise_gtfs_data(stops, stop_times)

            logger.info(f"Successfully loaded and normalised GTFS data for mode: {mode}")
            return stops, stop_times, routes, trips, calendar
        except FileNotFoundError as e:
            logger.error(f"File not found for mode '{mode}' in path '{path}': {e}")
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data encountered for mode '{mode}' in path '{path}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error while loading GTFS data for mode '{mode}': {e}")

        # Return None in case of any error
        return None

    @staticmethod
    def load_combined_data( self, train_path: str, tram_path: str, bus_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Author: AlexT
        Load and combine GTFS data for train, tram, and bus.
        """
        train_data = load_mode_data(train_path, "train")
        tram_data = load_mode_data(tram_path, "tram")
        bus_data = load_mode_data(bus_path, "bus")

        stops = pd.concat([train_data[0], tram_data[0], bus_data[0]], ignore_index=True)
        stop_times = pd.concat([train_data[1], tram_data[1], bus_data[1]], ignore_index=True)
        routes = pd.concat([train_data[2], tram_data[2], bus_data[2]], ignore_index=True)
        trips = pd.concat([train_data[3], tram_data[3], bus_data[3]], ignore_index=True)
        calendar = pd.concat([train_data[4], tram_data[4], bus_data[4]], ignore_index=True)

        return stops, stop_times, routes, trips, calendar

    @staticmethod
    def find_station_name(user_input: str, stops_df: pd.DataFrame) -> Optional[str]:
        """
            Author: AlexT
            Find the best matching station name from the stops DataFrame.
        """
        user_input = user_input.lower().strip()
        stops_df['word_count'] = stops_df['normalized_stop_name'].apply(lambda x: len(x.split()))

        exact_match = stops_df[stops_df['normalized_stop_name'] == user_input]
        if not exact_match.empty:
            return exact_match.iloc[0]['stop_name']

        keyword_matches = stops_df[stops_df['normalized_stop_name'].str.contains(user_input, na=False)].copy()

        if not keyword_matches.empty:
            keyword_matches['match_score'] = keyword_matches['normalized_stop_name'].apply(
                lambda name: sum(name.count(word) for word in user_input.split())
            )
            keyword_matches = keyword_matches.sort_values(by=['match_score', 'word_count'], ascending=[False, True])
            return keyword_matches.iloc[0]['stop_name']

        best_match, score, _ = process.extractOne(user_input, stops_df['stop_name'])
        if score > 80:
            return best_match

        return None

    @staticmethod
    def extract_stations_from_query(query: str, stops_df: pd.DataFrame) -> List[str]:
        """
            Author: AlexT
            Extract potential station names from a query using NLP and fuzzy matching.
        """
        doc = nlp(query)
        potential_stations = [ent.text for ent in doc.ents]
        print(f"Potential Stations (SpaCy): {potential_stations}")
        if not potential_stations:
            potential_stations = [GTFSUtils.find_station_name(query, stops_df)]

        extracted_stations = []
        for station in potential_stations:
            matched_station = GTFSUtils.find_station_name(station, stops_df)
            if matched_station:
                extracted_stations.append(matched_station)

        print(f"Extracted stations: {extracted_stations}")
        return extracted_stations

    @staticmethod
    def get_station_id(station_name: str, stops_df: pd.DataFrame) -> Optional[str]:
        """
            Author: AlexT
            Get the stop ID for a given station name, using fuzzy matching to find the correct station name.
        """
        matched_station_name = GTFSUtils.find_station_name(station_name, stops_df)
        if matched_station_name:
            station_row = stops_df.loc[stops_df['stop_name'] == matched_station_name]
            if not station_row.empty:
                return station_row['stop_id'].values[0]
        logger.error(f"Station name {station_name} not found in stops_df.")
        return None

    @staticmethod
    def find_common_routes(self, stop_a_id: str, stop_b_id: str) -> List[str]:
        """
        Author: AlexT
        Find common routes between two stops.
        """
        try:
            # Get trips passing through each stop
            trips_a = bus_stop_times.xs(key=stop_a_id, level='stop_id').index.get_level_values('trip_id').unique()
            trips_b = bus_stop_times.xs(key=stop_b_id, level='stop_id').index.get_level_values('trip_id').unique()

            # Find common trips
            common_trips = set(trips_a).intersection(trips_b)

            # Map trips to routes
            routes = bus_trips[bus_trips['trip_id'].isin(common_trips)]['route_id'].unique()
            return routes.tolist()
        except KeyError:
            return []
    @staticmethod
    def check_direct_route(station_a: str, station_b: str, stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> (
    bool, List[str]):
        """
            Author: AlexT
            Check if there is a direct train between two stations.
        """
        stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
        stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

        try:
            stop_a_times = stop_times_df.xs(stop_a_id, level='stop_id')
            stop_b_times = stop_times_df.xs(stop_b_id, level='stop_id')
        except KeyError:
            return False, []

        merged = pd.merge(stop_a_times.reset_index(), stop_b_times.reset_index(), on='trip_id', suffixes=('_a', '_b'))
        valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']]

        if not valid_trips.empty:
            return True, valid_trips['trip_id'].unique()
        return False, []

    @staticmethod
    def calculate_route_travel_time(route: List[str], stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> Optional[float]:
        """
            Author: AlexT
            Calculate the total travel time for a given route.
        """
        total_travel_time = 0.0

        stop_times_df.sort_index(inplace=True)

        for i in range(len(route) - 1):
            station_a = route[i]
            station_b = route[i + 1]

            direct_route_exists, trip_ids = GTFSUtils.check_direct_route(station_a, station_b, stops_df, stop_times_df)
            if not direct_route_exists:
                return None

            best_trip_id = trip_ids[0]
            stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

            try:
                stop_a_time = stop_times_df.loc[(stop_a_id, best_trip_id), 'departure_time']
                stop_b_time = stop_times_df.loc[(stop_b_id, best_trip_id), 'arrival_time']

                if isinstance(stop_a_time, pd.Series):
                    stop_a_time = stop_a_time.iloc[0]
                if isinstance(stop_b_time, pd.Series):
                    stop_b_time = stop_b_time.iloc[0]

            except KeyError:
                return None

            travel_time = GTFSUtils.parse_time(stop_b_time) - GTFSUtils.parse_time(stop_a_time)
            total_travel_time += travel_time.total_seconds()

        return total_travel_time / 60  # Return time in minutes

    @staticmethod
    def parse_time(gtfs_time: str) -> timedelta:
        """
            Author: AlexT
            Parse GTFS time (handling times that exceed 24:00:00) into a timedelta object.
        """
        hours, minutes, seconds = map(int, gtfs_time.split(':'))
        if hours >= 24:
            hours = hours - 24
            return timedelta(days=1, hours=hours, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def calculate_transfers(station_a: str, station_b: str, stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> \
            Tuple[int, List[str]]:
        """
            Author: AlexT
            Calculate the number of transfers needed between two stations and provide details of the transfer stations.
        """

        # Ensure the DataFrame is sorted and indexed correctly
        stop_times_df.sort_index(inplace=True)

        # Check for a direct route first
        direct_route_exists, _ = GTFSUtils.check_direct_route(station_a, station_b, stops_df, stop_times_df)
        if direct_route_exists:
            return 0, []  # No transfers needed, no transfer points

        # Get stop IDs for both stations
        stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
        stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

        if stop_a_id is None or stop_b_id is None:
            return -1, []  # Indicates that one of the stations could not be found

        # Initialize variables
        transfers = 0
        transfer_points = []
        visited_trips = set()

        try:
            # Extract the subset of stop_times for stop_a_id
            stop_a_times = stop_times_df.xs(key=stop_a_id, level='stop_id')
            possible_trips = stop_a_times.index.get_level_values('trip_id').unique()

            for trip_id in possible_trips:
                if trip_id in visited_trips:
                    continue

                visited_trips.add(trip_id)
                # Get all stops on the current trip
                trip_stops = stop_times_df.xs(key=trip_id, level='trip_id').index.get_level_values('stop_id').unique()

                if stop_b_id in trip_stops:
                    # If station_b is in the same trip, no transfer is needed
                    return transfers, transfer_points

            # If no direct trip found, increment transfers
            transfers += 1

            # Now find possible transfers
            for trip_id in possible_trips:
                trip_stops = stop_times_df.xs(key=trip_id, level='trip_id').index.get_level_values('stop_id').unique()

                for stop_id in trip_stops:
                    if stop_id != stop_a_id and stop_id != stop_b_id:
                        connecting_trip_times = stop_times_df.xs(key=stop_id, level='stop_id')
                        connecting_trips = connecting_trip_times.index.get_level_values('trip_id').unique()

                        for connecting_trip in connecting_trips:
                            if connecting_trip in visited_trips:
                                continue
                            visited_trips.add(connecting_trip)
                            connecting_trip_stops = stop_times_df.xs(key=connecting_trip,
                                                                     level='trip_id').index.get_level_values(
                                'stop_id').unique()

                            if stop_b_id in connecting_trip_stops:
                                # Record the transfer point
                                transfer_station = stops_df.loc[stops_df['stop_id'] == stop_id, 'stop_name'].values[0]
                                transfer_points.append(transfer_station)
                                return transfers, transfer_points
        except KeyError as e:
            print(f"KeyError when calculating transfers: {e}")
            return -1, []

        # If no route was found, assume infinite transfers
        return float('inf'), []
    @staticmethod
    def find_best_route_with_transfers(station_a: str, station_b: str, stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> Optional[List[str]]:
        """
            Author: AlexT
            Find the best route between two stations, considering transfers.
        """
        stop_times_df.sort_index(inplace=True)

        queue = deque([(station_a, [station_a])])
        visited = set()

        stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
        stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

        if stop_a_id is None or stop_b_id is None:
            return None

        while queue:
            current_station, path = queue.popleft()

            if current_station in visited:
                continue

            visited.add(current_station)

            direct_route_exists, trip_ids = GTFSUtils.check_direct_route(current_station, station_b, stops_df, stop_times_df)
            if direct_route_exists:
                return path + [station_b]

            current_stop_id = GTFSUtils.get_station_id(current_station, stops_df)
            if current_stop_id is None:
                continue

            try:
                stop_times_subset = stop_times_df.xs(key=current_stop_id, level='stop_id')
                possible_trips = stop_times_subset.index.get_level_values('trip_id').unique()

                for trip_id in possible_trips:
                    trip_stops = stop_times_df.xs(key=trip_id, level='trip_id')['stop_sequence'].index.get_level_values('stop_id').unique()
                    for next_stop_id in trip_stops:
                        next_station = stops_df.loc[stops_df['stop_id'] == next_stop_id, 'stop_name'].values[0]
                        if next_station not in visited:
                            queue.append((next_station, path + [next_station]))
            except KeyError:
                continue

        return None

    @staticmethod
    def handle_error(dispatcher, logger, message, error):
        error_message = f"{message}: {str(error)}"
        dispatcher.utter_message(text=error_message)
        logger.error(error_message)
        return []

    @staticmethod
    def get_trip_id_for_best_route(best_route: List[str], stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> \
    Optional[str]:
        """
            Author: AlexT
            Given a list of stops representing the best route, find the trip ID that matches this route.
            :param best_route: A list of stop names (or IDs) representing the best route.
            :param stops_df: DataFrame containing stop information.
            :param stop_times_df: DataFrame containing stop times.
            :return: The trip ID that best matches the provided route, or None if no suitable trip is found.
        """
        try:
            # Get the stop IDs for each station in the best route
            stop_ids = [GTFSUtils.get_station_id(station, stops_df) for station in best_route]

            if None in stop_ids:
                raise ValueError("One or more station IDs could not be found for the provided route.")

            # Filter stop_times_df for trips that pass through all stops in the best route
            matching_trips = stop_times_df.loc[stop_ids].reset_index()
            matching_trips = matching_trips.groupby('trip_id').filter(
                lambda x: set(stop_ids).issubset(x['stop_id'].tolist()))

            if matching_trips.empty:
                return None

            # Select the trip ID that has the stops in the correct order
            for trip_id, group in matching_trips.groupby('trip_id'):
                ordered_stop_ids = group.sort_values('stop_sequence')['stop_id'].tolist()
                if ordered_stop_ids == stop_ids:
                    return trip_id

            return None

        except Exception as e:
            logger.error(f"Failed to get trip ID for best route: {str(e)}")
            return None

    @staticmethod
    def is_subsequence(subsequence: List[str], sequence: List[str]) -> bool:
        """
            Author: AlexT
            Check if subsequence is a subsequence of sequence.
            :param subsequence: The list of stop IDs that form the best route.
            :param sequence: The list of stop IDs from a candidate trip.
            :return: True if subsequence is a subsequence of sequence, False otherwise.
        """
        iter_seq = iter(sequence)
        return all(item in iter_seq for item in subsequence)

    @staticmethod
    def generate_route_map(trip_id: str, station_a: str, station_b: str, stops_df: pd.DataFrame,
                           stop_times_df: pd.DataFrame, dataset_path: str,
                           transfers_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """
            Author: AlexT
            Generate a route map for a given trip ID, save it as an HTML file, and return a hyperlink to the map.
            Optionally highlights transfer stations on the route if transfers_df is provided.
            :param trip_id: The trip ID for which to generate the route map.
            :param station_a: The starting station name.
            :param station_b: The ending station name.
            :param stops_df: DataFrame containing stop information.
            :param stop_times_df: DataFrame containing stop times.
            :param dataset_path: Path to the dataset directory where the map will be saved.
            :param transfers_df: Optional DataFrame containing transfer information.
            :return: Hyperlink to the saved map file, or None if the map could not be generated.
        """
        try:
            # Ensure the stop_times_df is correctly indexed (it should already be normalized)
            if not isinstance(stop_times_df.index, pd.MultiIndex):
                raise ValueError("The stop_times_df DataFrame must be indexed by ['stop_id', 'trip_id'].")

            # Normalize the input station names to match the normalized stop names in stops_df
            normalized_station_a = station_a.strip().lower()
            normalized_station_b = station_b.strip().lower()

            # Get stop IDs for station_a and station_b using the normalized names
            stop_a_id = stops_df.loc[stops_df['normalized_stop_name'] == normalized_station_a, 'stop_id'].values[0]
            stop_b_id = stops_df.loc[stops_df['normalized_stop_name'] == normalized_station_b, 'stop_id'].values[0]

            # Filter the stop_times_df by the specified trip_id
            trip_stops = stop_times_df.xs(trip_id, level='trip_id').reset_index()

            if trip_stops.empty:
                raise ValueError(f"No stops found for trip ID {trip_id}.")

            # Find the sequences for station_a and station_b
            stop_a_sequence = trip_stops[trip_stops['stop_id'] == stop_a_id]['stop_sequence'].values[0]
            stop_b_sequence = trip_stops[trip_stops['stop_id'] == stop_b_id]['stop_sequence'].values[0]

            # Filter the trip stops to only include those between stop_a_sequence and stop_b_sequence
            trip_stops = trip_stops[
                (trip_stops['stop_sequence'] >= stop_a_sequence) & (trip_stops['stop_sequence'] <= stop_b_sequence)]

            # Merge with stops_df to get stop names and locations
            trip_stops = trip_stops.merge(stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']], on='stop_id',
                                          how='left')

            if trip_stops[['stop_lat', 'stop_lon']].isnull().any().any():
                raise ValueError(f"Missing stop latitude/longitude information for trip ID {trip_id}.")

            # Create the map centered around the first stop
            start_location = trip_stops[['stop_lat', 'stop_lon']].iloc[0].values
            trip_map = folium.Map(location=start_location, zoom_start=13)

            # Highlight transfer stations if transfers_df is provided
            if transfers_df is not None:
                transfer_stop_ids = transfers_df['from_stop_id'].unique()
            else:
                transfer_stop_ids = []

            # Plot each stop on the map, highlighting transfer stations in red
            for _, stop in trip_stops.iterrows():
                color = 'red' if stop['stop_id'] in transfer_stop_ids else 'blue'
                folium.Marker(
                    location=[stop['stop_lat'], stop['stop_lon']],
                    popup=f"Stop: {stop['stop_name']}<br>Stop ID: {stop['stop_id']}",
                    tooltip=stop['stop_name'],
                    icon=folium.Icon(color=color)
                ).add_to(trip_map)

            # Draw lines between consecutive stops to represent the route
            folium.PolyLine(
                locations=trip_stops[['stop_lat', 'stop_lon']].values.tolist(),
                color='blue',
                weight=5,
                opacity=0.7
            ).add_to(trip_map)

            # Save the map to an HTML file
            map_filename = f"route_map_{trip_id}.html"
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True) # create maps folder if doesn't exist
            map_path = os.path.join(map_folder, map_filename)
            trip_map.save(map_path)

            # Create and return the hyperlink
            # hyperlink = f"<a href='{map_path}' target='_blank'>Click here to view the route map</a>"
            # return hyperlink

            # Get the base URL from the environment variable
            server_base_url = os.getenv('SERVER_BASE_URL')

            # Fallback if the environment variable is not set
            if server_base_url is None:
                server_base_url = 'http://localhost:8080'  # Default value or fallback

            # Create and return the hyperlink using the base URL
            public_url = f"{server_base_url}/maps/{map_filename}"
            hyperlink = f"<a href='{public_url}' target='_blank'>Click here to view the route map</a>"
            return hyperlink


        except Exception as e:
            logger.error(f"Failed to generate route map for trip ID {trip_id}: {str(e)}")
            return None

    def generate_map(stops_df: pd.DataFrame, map_title: str, dataset_path: str) -> str:
        """
        Author: AlexT
        Generates a map for a given mode of transport and saves it as an HTML file.
        Args:
            stops_df (pd.DataFrame): DataFrame containing stop information (stop_id, stop_name, stop_lat, stop_lon).
            map_title (str): Title for the map (e.g., "Train Stations", "Tram Stops").
            dataset_path (str): Path where the generated map will be saved.
        Returns:
            str: File path to the generated map HTML.
        """
        try:
            # Extract necessary columns
            stops_map_df = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]

            # Initialize the map centered at Melbourne
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

            # Add markers for each stop
            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row['stop_lat'], row['stop_lon']],
                    popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
                    tooltip=row['stop_name']
                ).add_to(melbourne_map)

            # Save the map to an HTML file
            map_filename = f"{map_title.replace(' ', '_').lower()}_map.html"
            map_folder = os.path.join(dataset_path, "maps")
            os.makedirs(map_folder, exist_ok=True)  # Create the maps folder if it doesn't exist
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            return map_path

        except Exception as e:
            logging.error(f"Error generating {map_title} map: {e}")
            return None

    @staticmethod
    def generate_signature(base_url, user_id, api_key, route_id):
        """
        Author: AlexT
        Generate API signature.
        """
        url_path = f"/v3/disruptions/route/{route_id}"
        query_string = f"devid={user_id}"
        full_url = f"{base_url}{url_path}?{query_string}"

        parsed_url = urllib.parse.urlparse(full_url)
        url_to_sign = parsed_url.path + "?" + parsed_url.query
        signature = hmac.new(api_key.encode(), url_to_sign.encode(), hashlib.sha1).hexdigest()

        return f"{full_url}&signature={signature}"

    @staticmethod
    def fetch_disruptions(signed_url):
        """
        Author: AlexT
        Fetch disruptions from the API and ensure all transport modes (tram, bus, train) are handled.
        """
        try:
            response = requests.get(signed_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error while fetching disruptions: {e}")
            # Return an empty structure for all modes to ensure consistency
            return {"disruptions": {"metro_tram": [], "metro_bus": [], "metro_train": []}}
        except Exception as e:
            print(f"Unexpected error while fetching disruptions: {e}")
            # Handle other exceptions gracefully
            return {"disruptions": {"metro_tram": [], "metro_bus": [], "metro_train": []}}

    @staticmethod
    def filter_active_disruptions(disruptions):
        """
        Author: AlexT
        Filter currently active disruptions.
        """
        current_time = datetime.utcnow()
        active_disruptions = [
            d for d in disruptions
            if datetime.fromisoformat(d["from_date"].replace("Z", "")) <= current_time <= datetime.fromisoformat(
                d["to_date"].replace("Z", ""))
        ]
        return active_disruptions

    @staticmethod
    def check_route_and_fetch_disruptions(route_name, mode, routes_df):
        """
        Author: AlexT
        Check the route and fetch disruptions for tram, bus, or train.
        """
        # Match the route in the provided routes DataFrame
        matched_routes = routes_df[
            (routes_df["route_short_name"] == route_name) | (routes_df["route_long_name"] == route_name)
            ]

        if matched_routes.empty:
            return None, None, f"No routes found for '{route_name}'. Please check your input."

        route_id = matched_routes.iloc[0]["route_id"]
        signed_url = GTFSUtils.generate_signature(base_url, user_id, api_key, route_id)
        disruptions_data = GTFSUtils.fetch_disruptions(signed_url)

        print(f"check_route_and_fetch_disruptions MODE: {mode}")

        # Fetch disruptions based on the mode
        if mode == "tram":
            disruptions = disruptions_data.get("disruptions", {}).get("metro_tram", [])
        elif mode == "bus":
            disruptions = disruptions_data.get("disruptions", {}).get("metro_bus", [])
        elif mode == "train":
            disruptions = disruptions_data.get("disruptions", {}).get("metro_train", [])
        else:
            return None, None, f"Invalid mode: {mode}. Supported modes are 'tram', 'bus', and 'train'."

        # Filter active disruptions
        active_disruptions = GTFSUtils.filter_active_disruptions(disruptions)
        return active_disruptions, route_id, None


    @staticmethod
    def extract_route_name(query: str, routes_df: pd.DataFrame) -> Optional[str]:
        """
        Author: AlexT -- Working version for long name and short name
        Extract the route short name or validate the route long name from a user query.
        :param query: The user's query as a string.
        :param routes_df: DataFrame containing route information with 'route_short_name' and 'route_long_name'.
        :return: The extracted route short name if valid, or None if no match is found.
        """
        try:
            # Normalise query for consistent matching
            if not isinstance(query, str):
                print("Error: Query is not a string.")
                return None

            # Lowercase and normalize the query for matching
            query = query.lower().strip().replace(" to ", " - ")
            print(f"Normalised Query: {query}")

            # Normalize the DataFrame for comparison
            routes_df["route_short_name"] = routes_df["route_short_name"].astype(str).str.strip().str.lower()
            routes_df["route_long_name"] = routes_df["route_long_name"].astype(str).str.strip().str.lower()

            # Convert columns to lists for matching
            route_short_names = routes_df["route_short_name"].tolist()
            route_long_names = routes_df["route_long_name"].tolist()

            print(f"Available route short names: {route_short_names}")
            print(f"Available route long names: {route_long_names}")

            # Check if a route short name matches directly in the query
            for short_name in route_short_names:
                if short_name in query:
                    print(f"Direct match found for route_short_name: {short_name}")
                    return short_name

            # Check for route long names in the query
            for long_name in route_long_names:
                if isinstance(long_name, str) and long_name in query:
                    print(f"Direct match found for route_long_name: {long_name}")
                    matching_short_name = \
                        routes_df.loc[routes_df["route_long_name"] == long_name, "route_short_name"].iloc[0]
                    return matching_short_name

            # Use NLP to extract numerical tokens or keywords
            print("Attempting to extract using NLP...")
            doc = nlp(query)
            for token in doc:
                if token.text.lower() in route_short_names:
                    print(f"NLP extracted route_short_name: {token.text}")
                    return token.text.lower()

            # Attempt fuzzy matching for partial matches
            print("Attempting fuzzy matching...")
            matched_long_name, score = process.extractOne(
                query, route_long_names, scorer=fuzz.partial_ratio
            )
            if score > 85:  # Define a threshold for fuzzy matches
                print(f"Fuzzy match found: {matched_long_name} with score {score}")
                matched_index = route_long_names.index(matched_long_name)
                return route_short_names[matched_index]

            print("No match found for the query.")
            return None

        except Exception as e:
            print(f"An error occurred in extract_route_name: {str(e)}")
            return None

    @staticmethod
    def find_relevant_trips(stop_a_id, stop_b_id, trips_df, stop_times_df):
        """
        Author: AlexT
        Find trips that pass through both stops and occur in the correct sequence for Tram.
        """
        # Get trips serving stop_a
        stop_a_times = stop_times_df[stop_times_df['stop_id'] == stop_a_id]
        stop_b_times = stop_times_df[stop_times_df['stop_id'] == stop_b_id]

        # Merge to find trips that include both stops
        merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))

        # Filter trips where stop_a comes before stop_b
        valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']]

        # Filter for future trips only
        current_time = datetime.now().strftime('%H:%M:%S')
        valid_trips = valid_trips[valid_trips['departure_time_a'] >= current_time]

        # Join trip details
        valid_trips = valid_trips.merge(trips_df, on='trip_id')

        return valid_trips.sort_values('departure_time_a')
    @staticmethod
    def determine_user_route(
            query: str, stops_df: pd.DataFrame, bus_routes_df: pd.DataFrame, tram_routes_df: pd.DataFrame
    ) -> Optional[str]:
        """
        Author: AlexT
        Determine the route (bus or tram) the user is asking for based on their query.
        :param query: User-provided natural language query.
        :param stops_df: DataFrame containing stop names and normalized stop names.
        :param bus_routes_df: DataFrame containing bus routes.
        :param tram_routes_df: DataFrame containing tram routes.
        :return: Route short name, or None if no route matches the query.
        """
        try:
            # Normalize query
            query = query.lower().strip()

            # Step 1: Extract named entities using Hugging Face NER
            ner_pipeline = pipeline("ner", grouped_entities=True)
            entities = ner_pipeline(query)
            potential_stations = [entity['word'] for entity in entities if entity['entity_group'] == "LOC"]
            print(f"Potential Stations (Hugging Face NER): {potential_stations}")

            # Fallback to manual segmentation if needed
            if len(potential_stations) < 2:
                start_keywords = ["from", "starting at", "departing from", "leaving"]
                destination_keywords = ["to", "heading to", "going to", "destination"]

                def extract_segment(query, keywords, stop_at_keywords):
                    for keyword in keywords:
                        if keyword in query:
                            segment = query.split(keyword, 1)[-1].strip()
                            for stop_keyword in stop_at_keywords:
                                stop_idx = segment.find(stop_keyword)
                                if stop_idx != -1:
                                    segment = segment[:stop_idx].strip()
                            return segment.strip("?").strip()
                    return None

                start_station = extract_segment(query, start_keywords, destination_keywords)
                destination_station = extract_segment(query, destination_keywords, [])
                potential_stations = [start_station, destination_station]
                print(f"Fallback Stations: Start: {start_station}, Destination: {destination_station}")

            # Normalize potential station names for matching
            normalized_stations = [station.lower().strip() if station else None for station in potential_stations]

            # Determine the mode (bus or tram)
            if "tram" in query:
                routes_df = tram_routes_df
            elif "bus" in query:
                routes_df = bus_routes_df
            else:
                print("No mode (bus or tram) specified in query.")
                return None

            # Step 2: Attempt to match route short name directly
            for short_name in routes_df['route_short_name']:
                if f"route {short_name.lower()}" in query or short_name.lower() in query:
                    return short_name

            # Step 3: Match route long names using normalized station names
            start_station = normalized_stations[0]
            destination_station = normalized_stations[1]
            matching_routes = routes_df[
                routes_df['route_long_name'].str.contains(start_station or "", case=False, na=False) |
                routes_df['route_long_name'].str.contains(destination_station or "", case=False, na=False)
                ]

            if not matching_routes.empty:
                return matching_routes["route_short_name"].iloc[0]  # Return the first match

            print("No matching routes found.")
            return None

        except Exception as e:
            print(f"Error processing query: {query} | Exception: {e}")
            return None

    @staticmethod
    def determine_schedule(
            query: str,
            stops_df: pd.DataFrame,
            bus_routes_df: pd.DataFrame,
            tram_routes_df: pd.DataFrame,
            stop_times_df: pd.DataFrame,
            current_time: str
    ) -> Optional[str]:
        """
        Author: AlexT
        Determine the schedule for a specific route or stop based on the query.
        :param query: User-provided natural language query.
        :param stops_df: DataFrame containing stops data.
        :param bus_routes_df: DataFrame containing bus routes.
        :param tram_routes_df: DataFrame containing tram routes.
        :param stop_times_df: DataFrame containing stop times data.
        :param current_time: The current time in '%H:%M:%S' format.
        :return: A response string containing the schedule information.
        """
        try:
            # Step 1: Determine the route or mode using the refined determine_user_route
            route_short_name = GTFSUtils.determine_user_route(query, stops_df, bus_routes_df, tram_routes_df)

            # Step 2: Handle schedules for a route or a stop
            if route_short_name:
                # Case 1: Schedule for a specific route
                matching_trips = stop_times_df[stop_times_df['trip_id'].str.contains(route_short_name, na=False)]
                upcoming_trips = matching_trips[matching_trips['departure_time'] >= current_time].sort_values(
                    'departure_time').head(5)

                if not upcoming_trips.empty:
                    response = f"Upcoming schedules for route {route_short_name}:\n"
                    for _, row in upcoming_trips.iterrows():
                        departure_time = GTFSUtils.parse_time(row['departure_time'])
                        response += f"- Vehicle at {(datetime.min + departure_time).strftime('%I:%M %p')} from stop {row['stop_id']}\n"
                else:
                    response = f"No upcoming trips found for route {route_short_name}."
                return response

            else:
                # Case 2: Extract stations from the query (no route explicitly found)
                extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)
                station_a = extracted_stations[0]
                station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

                if station_a:
                    # Schedule for a specific stop
                    stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
                    matching_trips = stop_times_df[stop_times_df['stop_id'] == stop_a_id]
                    upcoming_trips = matching_trips[matching_trips['departure_time'] >= current_time].sort_values(
                        'departure_time').head(5)

                    if not upcoming_trips.empty:
                        response = f"Upcoming schedules from {station_a}:\n"
                        for _, row in upcoming_trips.iterrows():
                            departure_time = GTFSUtils.parse_time(row['departure_time'])
                            response += f"- Vehicle at {(datetime.min + departure_time).strftime('%I:%M %p')}\n"
                    else:
                        response = f"No upcoming trips found from {station_a}."
                    return response

            return "Sorry, I couldn't determine the schedule based on your query. Please try again with more details."

        except Exception as e:
            print(f"Error processing schedule: {e}")
            return "An error occurred while determining the schedule. Please try again later."
# Ross Start Functions
    @staticmethod
    def getAddressLatLong(address: str):
        """
            Author: RossP
            Lookup lat and long of address supplied by user
            :param address
            
        """
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
        "address": address,
        "components": "country:AU|administrative_area:VIC",
        "bounds": "-38.433859,144.593741|-37.511274,145.512529",  # Melbourne bounding box
        "key": 'AIzaSyAuNbb-Ttqw62DYDQlu64CgLcJ-Xp3_1JA'
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
    
            if data['status'] == "OK":
                # Extract details
                result = data['results'][0]
                formatted_address = result['formatted_address']
                location = result['geometry']['location']  # Contains 'lat' and 'lng'
                return {
                    "formatted_address": formatted_address,
                    "latitude": location['lat'],
                    "longitude": location['lng']
                }
            else:
                return {"error": f"Geocoding API error: {data['status']}"}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    @staticmethod
    def checkDistancetoAllStation(lat: float, long: float):
        """
            Author: RossP
            Lookup lat and long of address supplied by user
            :param lat
            :param long
            
        """
        try:
            train_stops = pd.read_csv("mpt_data/2/stops.txt")

            # iter data and calculate Euclidiean Distance
            dist_hold = float('inf')
            
            for row in train_stops.itertuples():
                
                dist = ((row.stop_lat - lat)**2 + (row.stop_lon - long)**2)**0.5
                
                if dist < dist_hold:
                    dist_hold = dist
                    closest_station = row.stop_name
                
            if closest_station:    
                return {
                    "closest_station_name": closest_station
                }
            else:
                return {"error": "No Station found"}
        except FileNotFoundError:
            return {"error": "File 'stops.txt' not found"}
        except pd.errors.EmptyDataError:
            return {"error": "File 'stops.txt' is empty or invalid"}
        except Exception as e:
            return {"error": str(e)}
    @staticmethod
    def checkDistancetoAllTramsStops(lat: float, long: float):
        """
            Author: RossP
            Lookup lat and long of address supplied by user
            :param lat
            :param long
            
        """
        try:
            tram_stops = pd.read_csv("mpt_data/3/stops.txt")

            # iter data and calculate Euclidiean Distance
            dist_hold = float('inf')
            
            for row in tram_stops.itertuples():
                
                dist = ((row.stop_lat - lat)**2 + (row.stop_lon - long)**2)**0.5
                
                if dist < dist_hold:
                    dist_hold = dist
                    closest_station = row.stop_name
                
            if closest_station:    
                return {
                    "closest_station_name": closest_station
                }
            else:
                return {"error": "No Station found"}
        except FileNotFoundError:
            return {"error": "File 'stops.txt' not found"}
        except pd.errors.EmptyDataError:
            return {"error": "File 'stops.txt' is empty or invalid"}
        except Exception as e:
            return {"error": str(e)}
    @staticmethod
    def checkDistancetoAllBusStops(lat: float, long: float):
        """
            Author: RossP
            Lookup lat and long of address supplied by user
            :param lat
            :param long
            
        """
        try:
            bus_stops = pd.read_csv("mpt_data/4/stops.txt")

            # iter data and calculate Euclidiean Distance
            dist_hold = float('inf')
            
            for row in bus_stops.itertuples():
                
                dist = ((row.stop_lat - lat)**2 + (row.stop_lon - long)**2)**0.5
                
                if dist < dist_hold:
                    dist_hold = dist
                    closest_station = row.stop_name
                
            if closest_station:    
                return {
                    "closest_station_name": closest_station
                }
            else:
                return {"error": "No Station found"}
        except FileNotFoundError:
            return {"error": "File 'stops.txt' not found"}
        except pd.errors.EmptyDataError:
            return {"error": "File 'stops.txt' is empty or invalid"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def getListOfStationsWithin1k(loc , transport_mode):
        """
            Author: RossP
            Get list of stations within 900m of locations 
            :param loc
            :param transport mode
            
        """
        try:
            if transport_mode == 'train':
                stops = pd.read_csv("mpt_data/2/stops.txt")
            
            if transport_mode == 'bus':
                stops = pd.read_csv("mpt_data/4/stops.txt")
                
            if transport_mode == 'tram':
                stops = pd.read_csv("mpt_data/3/stops.txt")

            # iter data and calculate stops within 1 km of location
            dist_res = []
            
            for row in stops.itertuples():
                
                x = loc['latitude'] - row.stop_lat
                y = (loc['longitude'] - row.stop_lon) * math.cos(row.stop_lat)
                dist = 110.25 * math.sqrt(x*x + y*y)
                
                if dist <= 0.9:
                    dist_res.append({'lat': row.stop_lat, 'lon': row.stop_lon, 'stop': row.stop_name, 'dist': dist})
            
            #top_5_closest = sorted(dist_res, key=lambda x: x["dist"])[:5]
             
            if len(dist_res)> 0:    
                return dist_res
            else:
                return {"error": "No Station found within 1k"}
        except FileNotFoundError:
            return {"error": f"The stops file for {transport_mode} was not found"}
        except pd.errors.EmptyDataError:
            return {"error": f"The stops file for {transport_mode} is empty or invalid"}
        except Exception as e:
            return {"error": str(e)}

#Ross End Functions