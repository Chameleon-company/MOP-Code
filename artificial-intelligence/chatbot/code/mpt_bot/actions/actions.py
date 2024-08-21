import spacy
import folium
import os
import pandas as pd
import logging
from typing import Any, Text, Dict, List, Optional
from fuzzywuzzy import process, fuzz
from datetime import datetime, timedelta
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Tuple
import logging
from collections import deque

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='rasa.shared.utils.io')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Load GTFS data into DataFrames
dataset_path = os.getenv('GTFS_DATASET_PATH', 'C:/Users/Alex.Truong/PycharmProjects/pythonProject/MPT/ds/gtfs/2')
stops_df = pd.read_csv(f'{dataset_path}/stops.txt')
routes_df = pd.read_csv(f'{dataset_path}/routes.txt')
trips_df = pd.read_csv(f'{dataset_path}/trips.txt')
stop_times_df = pd.read_csv(f'{dataset_path}/stop_times.txt')
calendar_df = pd.read_csv(f'{dataset_path}/calendar.txt')

'''-------------------------------------------------------------------------------------------------------'''
''' -------------------------------------------------------------------------------------------------------	
	Singleton common methods can be use in actions
	Author: AlexT
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
    def normalise_gtfs_data(stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> None:
        """Normalize the stop names and ensure the stop_times DataFrame is indexed correctly."""
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
        """Find the best matching station name from the stops DataFrame."""
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
        """Get the stop ID for a given station name, using fuzzy matching to find the correct station name."""
        matched_station_name = GTFSUtils.find_station_name(station_name, stops_df)
        if matched_station_name:
            station_row = stops_df.loc[stops_df['stop_name'] == matched_station_name]
            if not station_row.empty:
                return station_row['stop_id'].values[0]
        logger.error(f"Station name {station_name} not found in stops_df.")
        return None
    @staticmethod
    def extract_stations_from_query(query: str, stops_df: pd.DataFrame) -> List[str]:
        """Extract potential station names from a query using NLP and fuzzy matching."""
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
        """Check if there is a direct train between two stations."""
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
        """Calculate the total travel time for a given route."""
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

    # @staticmethod
    # def parse_time(gtfs_time: str) -> timedelta:
    #     """Parse GTFS time (assuming GTFS time format is HH:MM:SS) into a timedelta object."""
    #     hours, minutes, seconds = map(int, gtfs_time.split(':'))
    #     return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def parse_time(gtfs_time: str) -> timedelta:
        """Parse GTFS time (handling times that exceed 24:00:00) into a timedelta object."""
        hours, minutes, seconds = map(int, gtfs_time.split(':'))
        if hours >= 24:
            hours = hours - 24
            return timedelta(days=1, hours=hours, minutes=minutes, seconds=seconds)
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def calculate_transfers(station_a: str, station_b: str, stops_df: pd.DataFrame, stop_times_df: pd.DataFrame) -> \
            Tuple[int, List[str]]:
        """Calculate the number of transfers needed between two stations and provide details of the transfer stations."""

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
        """Find the best route between two stations, considering transfers."""
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

# Normalise data
GTFSUtils.normalise_gtfs_data(stops_df, stop_times_df)

''' -------------------------------------------------------------------------------------------------------
	ID: REQ_02 implementation
	Name: Schedule Information
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
'''
class ActionFindNextTrain(Action):

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
	ID: REQ_13 
	Name: Generate Map of Train Stations
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
'''
class ActionGenerateMap(Action):

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

''' -------------------------------------------------------------------------------------------------------
	ID: REQ_01 implementation
	Name: Basic Route Planning
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
	"What is the best route from [Station A] to [Station B]?"
    "How do I get from [Station A] to [Station B]?"
    "Show me the fastest route from [Station A] to [Station B]."
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

            response = f"The best route from {station_a} to {station_b} is on route {route_name}, trip ID {best_trip['trip_id']}, taking approximately {best_trip['travel_time'] / 60:.2f} minutes."
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

            transfers, transfer_stations = GTFSUtils.calculate_transfers(station_a, station_b, stops_df, stop_times_df)
            travel_time = GTFSUtils.calculate_route_travel_time([station_a] + transfer_stations + [station_b], stops_df, stop_times_df)

            if transfers == 0:
                response = f"There is a direct train from {station_a} to {station_b}, so no transfers are needed."
                if travel_time is not None:
                    response += f" The total travel time is approximately {travel_time:.2f} minutes."
            elif transfers > 0:
                transfer_details = ', '.join(transfer_stations) if transfer_stations else "unknown locations"
                response = (
                    f"You will need to make {transfers} transfer(s) to get from {station_a} to {station_b}. "
                    f"The transfer(s) occur at the following station(s): {transfer_details}."
                )
                if travel_time is not None:
                    response += f" The total travel time is approximately {travel_time:.2f} minutes."
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
                # Call calculate_route_travel_time with the correct arguments
                travel_time = GTFSUtils.calculate_route_travel_time([station_a, station_b], stops_df, stop_times_df)

                response = f"Yes, there is a direct train from {station_a} to {station_b} on trip ID {best_direct_trip}.\n"
                if travel_time is not None:
                    response += f"Total travel time: {travel_time:.2f} minutes."
                else:
                    response += "Unable to calculate travel time for the direct route."
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
                else:
                    response += f"No suitable route with transfers found between {station_a} and {station_b}."

            dispatcher.utter_message(text=response)

        except Exception as e:
            handle_error(dispatcher, logger, "Failed to check direct route or find the best route with transfers", e)
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
