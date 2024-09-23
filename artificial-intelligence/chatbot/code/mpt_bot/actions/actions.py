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
from rasa_sdk.events import SlotSet
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Tuple
from collections import deque
import certifi

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='rasa.shared.utils.io')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

DATASET_PATH = "./mnt/metro_train_accessibility_cleaned.csv"
station_data = pd.read_csv(DATASET_PATH)
station_data['Station Name'] = station_data['Station Name'].str.strip().str.lower()

'''-------------------------------------------------------------------------------------------------------'''
''' -------------------------------------------------------------------------------------------------------	
	Singleton common methods can be use in actions	
		- normalise_gtfs_data
		- find_station_name
		- convert_gtfs_time
		- parse_time
		- get_station_id
		- extract_stations_from_query
		- check_direct_route
		- calculate_route_travel_time
		- calculate_transfers
		- find_best_route_with_transfers
		- handle_error: handle error and logging
	-------------------------------------------------------------------------------------------------------
'''
class GTFSUtils:
    @staticmethod
    def load_data(url: str, dataset_path: str, inner_zip_path: str = '2/google_transit.zip') -> Optional[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
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
            Normalize the stop names and ensure the stop_times DataFrame is indexed correctly.
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
            map_folder = os.path.join(dataset_path, "maps")
            map_path = os.path.join(map_folder, map_filename)
            trip_map.save(map_path)

            # Create and return the hyperlink
            hyperlink = f"<a href='{map_path}' target='_blank'>Click here to view the route map</a>"
            return hyperlink

        except Exception as e:
            logger.error(f"Failed to generate route map for trip ID {trip_id}: {str(e)}")
            return None


# Normalise data
# GTFSUtils.normalise_gtfs_data(stops_df, stop_times_df)

url = 'https://data.ptv.vic.gov.au/downloads/gtfs.zip'
current_directory = os.getcwd()
dataset_folder = 'mpt_data'
dataset_path = os.path.join(current_directory, dataset_folder)
stops_df, stop_times_df, routes_df, trips_df, calendar_df = GTFSUtils.load_data(url, dataset_folder)


class ActionGenerateMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_13
    	Name: Generate Map of Train Stations
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_generate_map"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            stops_map_df = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row['stop_lat'], row['stop_lon']],
                    popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
                    tooltip=row['stop_name']
                ).add_to(melbourne_map)

            map_path = os.path.join(dataset_path, 'melbourne_train_stations_map.html')
            melbourne_map.save(map_path)

            dispatcher.utter_message(
                text=f"The map of Melbourne train stations has been generated and saved to: {map_path}")
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to generate map", e)
            raise

class ActionFindNextTrain(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_02 implementation
    	Name: Schedule Information
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_next_train"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or (not station_b and "to" in query.lower()):
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_station_id(station_b, stops_df) if station_b else None

            current_time = datetime.now().strftime('%H:%M:%S')

            if not isinstance(stop_times_df.index, pd.MultiIndex):
                stop_times_df.set_index(['stop_id', 'trip_id'], inplace=True, drop=False)

            if not station_b:
                trips_from_station = stop_times_df.loc[stop_a_id]
                trips_from_station = trips_from_station[trips_from_station['departure_time'] >= current_time]
                trips_from_station = trips_from_station.sort_values('departure_time').drop_duplicates(
                    subset=['departure_time'])

                if not trips_from_station.empty:
                    next_trips = trips_from_station[['departure_time']].head(5)
                    response = f"Upcoming train schedules from {station_a}:\n"
                    for idx, row in next_trips.iterrows():
                        departure_time = GTFSUtils.parse_time(row['departure_time'])
                        response += f"- Train at {(datetime.min + departure_time).strftime('%I:%M %p')}\n"
                else:
                    response = f"No upcoming trains found from {station_a}."
            else:
                trips_from_station_a = stop_times_df.loc[stop_a_id].reset_index()
                trips_to_station_b = stop_times_df.loc[stop_b_id].reset_index()

                future_trips = trips_from_station_a[trips_from_station_a['departure_time'] >= current_time][
                    'trip_id'].unique()
                valid_trips = trips_to_station_b[trips_to_station_b['trip_id'].isin(future_trips)]

                if not valid_trips.empty:
                    next_trip = valid_trips.iloc[0]
                    next_trip_time = trips_from_station_a[
                        (trips_from_station_a['trip_id'] == next_trip['trip_id'])
                    ]['departure_time'].values[0]
                    next_trip_time = GTFSUtils.parse_time(next_trip_time)
                    if isinstance(next_trip_time, timedelta):
                        next_trip_time = (datetime.min + next_trip_time).strftime('%I:%M %p')
                    response = f"The next train from {station_a} to {station_b} leaves at {next_trip_time}."
                else:
                    response = f"No upcoming trains found from {station_a} to {station_b}."

            dispatcher.utter_message(text=response)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the next train", e)
            raise


''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_01 implementation
    	Name: Basic Route Planning
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
    	"What is the best route from [Station A] to [Station B]?"
        "How do I get from [Station A] to [Station B]?"
        "Show me the fastest route from [Station A] to [Station B]."
        Generate the route map
'''
class ActionFindBestRoute(Action):

    def name(self) -> Text:
        return "action_find_best_route"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or not station_b:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

            stop_a_times = stop_times_df.loc[stop_a_id][['stop_sequence', 'arrival_time']].reset_index()
            stop_b_times = stop_times_df.loc[stop_b_id][['stop_sequence', 'arrival_time']].reset_index()

            merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))

            valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()

            if valid_trips.empty:
                dispatcher.utter_message(text="No direct route found between the two stations.")
                return []

            valid_trips['arrival_time_a'] = valid_trips['arrival_time_a'].apply(GTFSUtils.parse_time)
            valid_trips['arrival_time_b'] = valid_trips['arrival_time_b'].apply(GTFSUtils.parse_time)
            valid_trips['travel_time'] = (
                        valid_trips['arrival_time_b'] - valid_trips['arrival_time_a']).dt.total_seconds()

            best_trip = valid_trips.loc[valid_trips['travel_time'].idxmin()]

            route_id = trips_df.loc[trips_df['trip_id'] == best_trip['trip_id'], 'route_id'].values[0]
            route_name = routes_df.loc[routes_df['route_id'] == route_id, 'route_long_name'].values[0]
            destination = trips_df.loc[trips_df['trip_id'] == best_trip['trip_id'], 'trip_headsign'].values[0]

            response = f"The best route from {station_a} to {station_b} is on the {route_name} towards {destination} \n The trip taking approximately {best_trip['travel_time'] / 60:.2f} minutes."

            # Create the route map given the trip id, including the transfers_df to highlight transfer stations
            hyperlink = GTFSUtils.generate_route_map(best_trip['trip_id'], station_a, station_b, stops_df, stop_times_df, dataset_path)
            if hyperlink:
                response += f"\n{hyperlink}"

            dispatcher.utter_message(text=response)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the best route", e)
            raise

''' -------------------------------------------------------------------------------------------------------
	ID: REQ_03 implementation
	Name: Basic Route Planning
	Author: AlexT
	-------------------------------------------------------------------------------------------------------	
    "How many transfers are there between [Station A] and [Station B]?"    
    -------------------------------------------------------------------------------------------------------
    Testing:
    When testing this refactored action, ensure you test various scenarios:    
    Transfer Route: Ensure all transfer stations and the correct travel time are included.
    No Route Found: Confirm that the message correctly informs the user when no route is available. recommend the best route with transfer
    Invalid Stations: Check how the action handles cases where stations are not found or the travel time cannot be calculated.
'''

class ActionCalculateTransfers(Action):

    def name(self) -> Text:
        return "action_calculate_transfers"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            # Calculate transfers and transfer stations
            transfers, transfer_stations = GTFSUtils.calculate_transfers(station_a, station_b, stops_df, stop_times_df)
            travel_time = GTFSUtils.calculate_route_travel_time([station_a] + transfer_stations + [station_b], stops_df,
                                                                stop_times_df)

            if transfers == 0:
                response = f"There is a direct train from {station_a} to {station_b}, so no transfers are needed."
                if travel_time is not None:
                    response += f" The total travel time is approximately {travel_time:.2f} minutes."

                # Generate the route map for the direct route
                # best_trip_id = GTFSUtils.find_best_trip_id(station_a, station_b, stop_times_df)
                # if best_trip_id:
                #     hyperlink = GTFSUtils.generate_route_map(best_trip_id, station_a, station_b, stops_df,
                #                                              stop_times_df, dataset_path)
                #     if hyperlink:
                #         response += f"\n{hyperlink}"

            elif transfers > 0:
                transfer_details = ', '.join(transfer_stations) if transfer_stations else "unknown locations"
                response = (
                    f"You will need to make {transfers} transfer(s) to get from {station_a} to {station_b}. "
                    f"The transfer(s) occur at the following station(s): {transfer_details}."
                )
                if travel_time is not None:
                    response += f" The total travel time is approximately {travel_time:.2f} minutes."

                # Generate the route map with transfers (if applicable)
                # best_trip_id = GTFSUtils.find_best_trip_id_with_transfers(station_a, station_b, transfer_stations,
                #                                                           stop_times_df)
                # if best_trip_id:
                #     hyperlink = GTFSUtils.generate_route_map(best_trip_id, station_a, station_b, stops_df,
                #                                              stop_times_df, dataset_path, transfers_df)
                #     if hyperlink:
                #         response += f"\n{hyperlink}"

            else:
                response = f"Sorry, no suitable route with transfers could be found between {station_a} and {station_b}."

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while calculating transfers. Please try again.")
            GTFSUtils.handle_error(dispatcher, logger, "Failed to calculate transfers", e)
        return []


''' -------------------------------------------------------------------------------------------------------
	ID: REQ_03 implementation
	Name: Check Direct Route
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
	"Is there a direct train from [Station A] to [Station B]?"   
    -------------------------------------------------------------------------------------------------------
    Direct Route: Ensure the travel time is correct and no transfer details are included.
    No Route Found: Confirm that the message correctly informs the user when no route is available. recommend the best route with transfer            
'''
class ActionCheckDirectRoute(Action):

    def name(self) -> Text:
        return "action_check_direct_route"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            # Check for a direct route first
            direct_route_exists, trips = GTFSUtils.check_direct_route(station_a, station_b, stops_df, stop_times_df)

            if direct_route_exists:
                # If a direct route exists, return the best direct route (next available train)
                best_direct_trip = trips[0]  # Assume the first trip is the best for simplicity
                travel_time = GTFSUtils.calculate_route_travel_time([station_a, station_b], stops_df, stop_times_df)

                response = f"Yes, there is a direct train from {station_a} to {station_b}."
                if travel_time is not None:
                    response += f" Total travel time: {travel_time:.2f} minutes."
                else:
                    response += " Unable to calculate travel time for the direct route."

                hyperlink = GTFSUtils.generate_route_map(best_direct_trip, station_a, station_b, stops_df, stop_times_df, dataset_path)
                if hyperlink:
                    response += f"\n{hyperlink}"

            else:
                # If no direct route exists, find the best route with transfers
                response = f"No direct route found between {station_a} and {station_b}.\n"
                best_route = GTFSUtils.find_best_route_with_transfers(station_a, station_b, stops_df, stop_times_df)
                if best_route:
                    response += f"Best route with transfers: {' -> '.join(best_route)}\n"
                    travel_time = GTFSUtils.calculate_route_travel_time(best_route, stops_df, stop_times_df)
                    if travel_time is not None:
                        response += f"Total travel time: {travel_time:.2f} minutes."
                    else:
                        response += "Unable to calculate travel time for the best route."

                    # Generate a route map for the best route
                    best_trip_id = GTFSUtils.get_trip_id_for_best_route(best_route, stops_df, stop_times_df)
                    if best_trip_id:
                        # Pass transfer stations to highlight them on the map
                        hyperlink = GTFSUtils.generate_route_map(best_trip_id, station_a, station_b, stops_df,
                                                                 stop_times_df, dataset_path,
                                                                 transfer_stations=best_route[1:-1])
                        if hyperlink:
                            response += f"\n{hyperlink}"
                else:
                    response += f"No suitable route with transfers found between {station_a} and {station_b}."

            dispatcher.utter_message(text=response)

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to check direct route or find the best route with transfers", e)
            raise


''' -------------------------------------------------------------------------------------------------------
	ID: REQ_03 implementation
	Name: Best route with transfer
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
	How many transfers are there between Dandenong and Parliament?
    How many transfers are there between South Yarra to Hawthorns?
    How many transfers are there between North Melbourne to Hawthorns?
    Do I need to change trains to get from Melbourne Central to Flinders station?   
    -------------------------------------------------------------------------------------------------------                
'''
class ActionFindBestRouteWithTransfers(Action):

    def name(self) -> Text:
        return "action_find_best_route_with_transfers"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            # Use the generic method from GTFSUtils to find the best route with transfers
            best_route = GTFSUtils.find_best_route_with_transfers(station_a, station_b, stops_df, stop_times_df)
            if not best_route:
                dispatcher.utter_message(text=f"Sorry, I couldn't find a suitable route from {station_a} to {station_b}.")
                return []

            travel_time = GTFSUtils.calculate_route_travel_time(best_route, stops_df, stop_times_df)
            response = f"The best route from {station_a} to {station_b} involves the following transfers: {', '.join(best_route)}.\n"
            if travel_time:
                response += f"The total travel time is approximately {travel_time:.2f} minutes."

            dispatcher.utter_message(text=response)

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the best route with transfers", e)
            raise

        return []

''' -------------------------------------------------------------------------------------------------------
	ID: REQ_03 implementation
	Name: Check Train Change
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
	"Do I need to change trains to get to [Station B] from [Station A]?"   
    -------------------------------------------------------------------------------------------------------                
'''
class ActionCheckTrainChange(Action):

    def name(self) -> Text:
        return "action_check_train_change"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            result = GTFSUtils.check_train_change(station_a, station_b, stops_df, stop_times_df)
            dispatcher.utter_message(text=result)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to check train change", e)
            raise

''' -------------------------------------------------------------------------------------------------------
	ID: REQ_04 implementation
	Name: Route Optimisation
	Author: AlexT
	-------------------------------------------------------------------------------------------------------	
    "Which route has the least number of stops from [Station A] to [Station B]?"
    "What is the quickest route to avoid delays?"
    "Recommend a route to avoid Zone [X]."   
    -------------------------------------------------------------------------------------------------------                
'''
class ActionFindRouteWithLeastStops(Action):

    def name(self) -> Text:
        return "action_find_route_with_least_stops"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1]

            # Ensure the data is normalized
            if 'normalized_stop_name' not in stops_df.columns:
                logger.error("The column 'normalized_stop_name' does not exist in stops_df. Ensure that the data has been normalized.")
                dispatcher.utter_message(text="Error: 'normalized_stop_name' column missing from stops_df.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_station_id(station_b, stops_df)

            if stop_a_id is None or stop_b_id is None:
                dispatcher.utter_message(text=f"Unable to find Station IDs for either {station_a} or {station_b}. Please check the station names.")
                return []

            try:
                stop_a_times = stop_times_df.loc[stop_a_id].reset_index()
                stop_b_times = stop_times_df.loc[stop_b_id].reset_index()
            except KeyError as e:
                logger.error(f"KeyError when accessing stop_times_df: {e}")
                dispatcher.utter_message(text="Error accessing stop_times_df.")
                return []

            merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))
            valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()

            if valid_trips.empty:
                dispatcher.utter_message(text=f"No direct route found between {station_a} and {station_b}.")
                return []

            valid_trips.loc[:, 'number_of_stops'] = valid_trips['stop_sequence_b'] - valid_trips['stop_sequence_a']
            least_stops_trip = valid_trips.loc[valid_trips['number_of_stops'].idxmin()]

            route_id = trips_df.loc[trips_df['trip_id'] == least_stops_trip['trip_id'], 'route_id'].values[0]
            route_name = routes_df.loc[routes_df['route_id'] == route_id, 'route_long_name'].values[0]

            response = f"The route with the least number of stops from {station_a} to {station_b} is on route {route_name}, trip ID {least_stops_trip['trip_id']}, with {least_stops_trip['number_of_stops']} stops."
            dispatcher.utter_message(text=response)

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the route with least stops", e)
            raise

        return []

class ActionGenerateRouteMap(Action):

    def name(self) -> Text:
        return "action_generate_route_map"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            # Find the best route with transfers first
            best_route = GTFSUtils.find_best_route_with_transfers(station_a, station_b, stops_df, stop_times_df)
            if not best_route:
                dispatcher.utter_message(text=f"Sorry, I couldn't find a suitable route from {station_a} to {station_b}.")
                return []

            # Get trip IDs from the best route
            trip_id = GTFSUtils.get_trip_id_for_route(best_route, stops_df, stop_times_df)
            if not trip_id:
                dispatcher.utter_message(text=f"Sorry, I couldn't determine the trip ID for the route.")
                return []

            # Generate the map
            trip_stops = stop_times_df[stop_times_df['trip_id'] == trip_id]
            first_stop_id = trip_stops.iloc[0]['stop_id']
            first_stop = stops_df[stops_df['stop_id'] == first_stop_id]
            start_coords = [first_stop['stop_lat'].values[0], first_stop['stop_lon'].values[0]]

            melbourne_map = folium.Map(location=start_coords, zoom_start=12)

            for _, stop_time in trip_stops.iterrows():
                stop_id = stop_time['stop_id']
                stop_info = stops_df[stops_df['stop_id'] == stop_id]
                stop_name = stop_info['stop_name'].values[0]
                stop_coords = [stop_info['stop_lat'].values[0], stop_info['stop_lon'].values[0]]

                folium.Marker(
                    location=stop_coords,
                    popup=f"{stop_name}",
                    tooltip=stop_name
                ).add_to(melbourne_map)

            coords = [[stop_info['stop_lat'].values[0], stop_info['stop_lon'].values[0]] for stop_id in trip_stops['stop_id']]
            folium.PolyLine(coords, color="blue", weight=2.5, opacity=1).add_to(melbourne_map)

            map_path = os.path.join(dataset_path, f'{trip_id}_route_map.html')
            melbourne_map.save(map_path)

            dispatcher.utter_message(text=f"The map of the route from {station_a} to {station_b} has been generated and saved to: {map_path}")
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to generate route map", e)
            raise

''' -------------------------------------------------------------------------------------------------------
	
	Name: Directions and Mapping
	Author: LoganG
	-------------------------------------------------------------------------------------------------------
'''
from typing import Any, Text, Dict, List
import subprocess
import sys

logger = logging.getLogger(__name__)

class ActionRunMappingScript(Action):
    def name(self) -> Text:
        return "action_run_direction_script"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #Get the two most recent user messages
        user_messages = [event['text'] for event in tracker.events if event.get('event') == 'user']
        user_location = user_messages[-2] if len(user_messages) >= 2 else None
        destination = user_messages[-1] if user_messages else None

        current_dir = os.path.dirname(os.path.abspath(__file__))

        if not user_location or not destination:
            dispatcher.utter_message(text="I couldn't understand the location or destination. Please provide both.")
            return []

        script_path = os.path.join(current_dir, "userlocationmaps_executablepassingactions.py")

        
        try:
            result = subprocess.run([sys.executable, script_path, user_location, destination], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            
            output = result.stdout.strip()

            if output:
                output_parts = output.split("|||")
                
                if len(output_parts) >= 2:
                    description = output_parts[0].strip()
                    map_file_path = output_parts[1].strip()

                    dispatcher.utter_message(text=description)

                    #Generate URL for the map file using the existing server
                    relative_path = os.path.relpath(map_file_path, current_dir)
                    map_url = f"http://localhost:8000/{relative_path.replace(os.sep, '/')}"

                    map_link = f"<a href='{map_url}' target='_blank'>Click here to view the route map</a>"
                    dispatcher.utter_message(text=f"I've generated a route map for you: {map_link}")
                else:
                    dispatcher.utter_message(text="The script returned incomplete output. Please try again.")

            else:
                dispatcher.utter_message(text="The direction script has been executed successfully, but no output was produced.")

        except subprocess.CalledProcessError as e:  
            error_message = e.stderr.strip() if e.stderr else "Unknown error occurred during script execution."
            dispatcher.utter_message(text=f"An error occurred while running the script: {error_message}")
            logger.error(f"Script execution failed: {error_message}")
        except Exception as e:
            dispatcher.utter_message(text=f"An unexpected error occurred: {str(e)}")
            logger.error(f"Exception occurred: {str(e)}")

        return [SlotSet("user_location", user_location), SlotSet("destination", destination)]

''' 
-------------------------------------------------------------------------------------------------------
Author: hariprasad
-------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
Class: ActionCheckFeature
Purpose: This class handles the intent where a user asks whether a specific feature is available at a particular station.
--------------------------------------------------------------------------------------------------------------------------------------------------
'''

class ActionCheckFeature(Action):

    def name(self) -> Text:
        return "action_check_feature"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        station_name = tracker.get_slot("station_name")
        feature = tracker.get_slot("feature")
        
        if not station_name or not feature:
            dispatcher.utter_message(text="Please specify both the station name and the feature you are asking about.")
            return []

        station_name = station_name.strip().lower()
        feature = feature.strip().lower()

        station_names = station_data['Station Name'].tolist()
        if station_name not in station_names:
            dispatcher.utter_message(text=f"Sorry, I don't have information about {station_name.capitalize()} station.")
            return []

        feature_mapping = {
            "escalators": "Escalators",
            "escalator": "Escalators",
            "lifts": "Lift",
            "elevator": "Lift",
            "elevators": "Lift",  
            "ramps": "Station access",
            "access": "Station access",
            "parking": "Parking",
            "restroom": "Toilet",
            "toilets": "Toilet",
            "toilet": "Toilet",
            "tactile edges": "Tactile edges",
            "hearing loops": "Hearing Loop",
            "info screens": "Info screens",
            "shelter": "Shelter",
            "low platform": "Low platform",
            "path widths": "Path Widths",
            "pick up / drop off": "Pick up / Drop off"
        }

        standardized_feature = feature_mapping.get(feature)

        if not standardized_feature:
            dispatcher.utter_message(text=f"Sorry, I don't have information about {feature}.")
            return []

        station_info = station_data[station_data['Station Name'] == station_name]
        
        feature_value = station_info[standardized_feature].values[0]
        if pd.isna(feature_value) or feature_value.lower() == 'no':
            dispatcher.utter_message(text=f"No, {station_name.capitalize()} station does not have {feature}.")
        else:
            dispatcher.utter_message(text=f"Yes, {station_name.capitalize()} station has {feature}.")
        
        return []

'''
-------------------------------------------------------------------------------------------------------
Class: ActionCheckStation
Purpose: This class is used when a user asks about all features available at a specific station.
-------------------------------------------------------------------------------------------------------
'''

class ActionCheckStation(Action):

    def name(self) -> Text:
        return "action_check_station"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        station_name = tracker.get_slot("station_name")
        station_name = station_name.strip().lower()

        if station_name not in station_data['Station Name'].tolist():
            dispatcher.utter_message(text=f"Sorry, I don't have information about {station_name} station.")
            return []

        station_info = station_data[station_data['Station Name'] == station_name]
        
        features = station_info.iloc[0].to_dict()
        feature_descriptions = "\n".join([f"{k}: {v}" for k, v in features.items() if k != "Station Name"])
        dispatcher.utter_message(text=f"Here are the accessibility features for {station_name.capitalize()} station:\n{feature_descriptions}")
        
        return []

'''
-------------------------------------------------------------------------------------------------------
Class: ActionListAllStations
Purpose: This class is responsible for listing all the stations available in the dataset.
-------------------------------------------------------------------------------------------------------
'''

class ActionListAllStations(Action):

    def name(self) -> Text:
        return "action_list_all_stations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        unique_stations = sorted(set(station_data['Station Name'].str.title()))

        stations_list = ", ".join(unique_stations)
        dispatcher.utter_message(text=f"Here are all the metro stations: {stations_list}.")

        return []

'''
-----------------------------------------------------------------------------------------------------------
Class: ActionListStationsWithFeature
Purpose: This class handles queries where the user wants to find all stations that have a specific feature.
-----------------------------------------------------------------------------------------------------------
'''

class ActionListStationsWithFeature(Action):

    def name(self) -> Text:
        return "action_list_stations_with_feature"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        feature = tracker.get_slot("feature").strip().lower()
        
        feature_mapping = {
            "escalators": "Escalators",
            "escalator": "Escalators",
            "lifts": "Lift",
            "elevator": "Lift",
            "elevators": "Lift",
            "ramps": "Station access",
            "access": "Station access",
            "parking": "Parking",
            "restroom": "Toilet",
            "toilets": "Toilet",
            "tactile edges": "Tactile edges",
            "hearing loops": "Hearing Loop",
            "info screens": "Info screens",
            "shelter": "Shelter",
            "low platform": "Low platform",
            "path widths": "Path Widths",
            "pick up / drop off": "Pick up / Drop off"
        }

        standardized_feature = feature_mapping.get(feature)

        if not standardized_feature:
            dispatcher.utter_message(text=f"Sorry, I don't have information about {feature}.")
            return []

        stations_with_feature = station_data[station_data[standardized_feature].str.lower() == 'yes']['Station Name'].str.title().tolist()

        if not stations_with_feature:
            dispatcher.utter_message(text=f"No stations have {feature}.")
        else:
            stations_list = ", ".join(stations_with_feature)
            dispatcher.utter_message(text=f"Stations with {feature}: {stations_list}")

        return []
