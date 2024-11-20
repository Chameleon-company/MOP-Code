'''-------------------------------------------------------------------------------------------------------'''
''' -------------------------------------------------------------------------------------------------------
	GTFSUtils Singleton common methods can be use in actions - This is for train station only
	    - download_and_extract_data: (Download the whole dataset from ptv) - Author: AlexT
	    - load_data: - Author: AlexT
	    - load_mode_data: (Load data for a specific mode - train, tram or bus etc.) - Author: AlexT
	    - load_combined_data: (load and combine data for train, tram and bus)- Author: AlexT
		- normalise_gtfs_data: - Author: AlexT
		- find_station_name: - Author: AlexT
		- convert_gtfs_time: - Author: AlexT
		- parse_time: - Author: AlexT
		- get_station_id: - Author: AlexT
		- extract_stations_from_query: - Author: AlexT
		- check_direct_route: - Author: AlexT
		- calculate_route_travel_time: - Author: AlexT
		- calculate_transfers: - Author: AlexT
		- find_best_route_with_transfers: - Author: AlexT
		- handle_error: handle error and logging: - Author: AlexT
	-------------------------------------------------------------------------------------------------------
'''
import spacy
import folium
import os
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

logger = logging.getLogger(__name__)
# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

class GTFSUtils:
    @staticmethod
    def load_data(url: str, dataset_path: str, inner_zip_path: str = '2/google_transit.zip') -> Optional[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Author: AlexT
        Downloads, extracts, and loads GTFS data from a provided URL.
        :param url: The URL to download the GTFS zip file from.
        :param dataset_path: The folder where the extracted data will be stored.
        :param inner_zip_path: The path within the main zip to the GTFS data zip file.
        :return: A tuple containing the stops DataFrame, stop_times DataFrame, routes DataFrame, trips DataFrame, and calendar DataFrame.
        """
        # Ensure the dataset path is absolute and create it if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        logger.info("Downloading the zip file...")
        # response = requests.get(url)
        response = requests.get(url, verify=certifi.where())

        # Disable SSL certificate verification to bypass the certificate error
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()  # Ensure the download was successful
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return None

        # Open the downloaded zip file in memory
        logger.info("Opening the main zip file...")
        try:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # List all files and directories in the main zip file
                logger.info("Listing contents of the main zip file...")
                for name in z.namelist():
                    logger.info(f"Found file: {name}")
                    if name == inner_zip_path:
                        logger.info(f"Found '{inner_zip_path}' inside the main zip file.")
                        # Extract the inner zip file (google_transit.zip)
                        with z.open(name) as inner_zip_file:
                            with zipfile.ZipFile(BytesIO(inner_zip_file.read())) as inner_z:
                                logger.info(f"Extracting contents to '{dataset_path}'...")
                                inner_z.extractall(dataset_path)
                                logger.info(f"Extraction complete. Files saved to '{dataset_path}'")
                                break
                else:
                    logger.error(f"No matching '{inner_zip_path}' file found inside the main zip file.")
                    return None
        except Exception as e:
            logger.error(f"Error extracting zip files: {e}")
            return None

        # Load the relevant files into DataFrames
        try:
            stops_df = pd.read_csv(os.path.join(dataset_path, 'stops.txt'))
            routes_df = pd.read_csv(os.path.join(dataset_path, 'routes.txt'))
            trips_df = pd.read_csv(os.path.join(dataset_path, 'trips.txt'))
            stop_times_df = pd.read_csv(os.path.join(dataset_path, 'stop_times.txt'))
            calendar_df = pd.read_csv(os.path.join(dataset_path, 'calendar.txt'))

            # Normalize the data
            GTFSUtils.normalise_gtfs_data(stops_df, stop_times_df)

            return stops_df, stop_times_df, routes_df, trips_df, calendar_df
        except FileNotFoundError as e:
            logger.error(f"Failed to load GTFS data files: {e}")
            return None
        except pd.errors.EmptyDataError as e:
            logger.error(f"File found but no data: {e}")
            return None

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
        if not potential_stations:
            potential_stations = [GTFSUtils.find_station_name(query, stops_df)]

        extracted_stations = []
        for station in potential_stations:
            matched_station = GTFSUtils.find_station_name(station, stops_df)
            if matched_station:
                extracted_stations.append(matched_station)
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
