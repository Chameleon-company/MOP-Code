import spacy
import folium
from folium.plugins import MarkerCluster
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
from actions.gtfs_utils import GTFSUtils
from sanic import Sanic
from sanic.response import text
from geopy.distance import geodesic
import hashlib
import hmac
import urllib.parse
from tabulate import tabulate
from rasa_sdk.events import SlotSet

# This is to skip the favicon
app = Sanic("custom_action_server")
@app.route("/favicon.ico")
async def favicon(request):
    return text("")  # Ignore the favicon request

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='rasa.shared.utils.io')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# ---------------------------------------------------------------------------------------------------------------------
# AlexT - Start Global Variables --------------------------------------------------------------------------------------
url = 'https://data.ptv.vic.gov.au/downloads/gtfs.zip'
current_directory = os.getcwd()
dataset_folder = 'mpt_data'
dataset_path = os.path.join(current_directory, dataset_folder)
inner_zip_paths = ['2/google_transit.zip', '3/google_transit.zip', '4/google_transit.zip']
# AlexT: Commented this out so you don't have to download the dataset every time you start the chat bot
GTFSUtils.download_and_extract_data(url, dataset_path, inner_zip_paths)

train_data = GTFSUtils.load_mode_data("mpt_data/2", "train")
tram_data = GTFSUtils.load_mode_data("mpt_data/3", "tram")
bus_data = GTFSUtils.load_mode_data("mpt_data/4", "bus")

# Unpack dataset for train
if train_data:
    stops_df, stop_times_df, routes_df, trips_df, calendar_df = train_data
else:
    logger.error("Failed to load Train data.")

# Unpack dataset for bus
if bus_data:
    bus_stops, bus_stop_times, bus_routes, bus_trips, bus_calendar = bus_data
else:
    logger.error("Failed to load Bus data.")

# Unpack dataset for tram
if tram_data:
    tram_stops, tram_stop_times, tram_routes, tram_trips, tram_calendar = tram_data
else:
    logger.error("Failed to load Tram data.")
# AlexT - End Global Variables --------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Hari - Start Global Variables --------------------------------------------------------------------------------------
CSV_DATASET_PATH = "./mnt/metro_train_accessibility_cleaned.csv"
station_data = pd.read_csv(CSV_DATASET_PATH)
station_data['Station Name'] = station_data['Station Name'].str.strip().str.lower()
# Hari - End Global Variables --------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

class ActionFindNextTram(Action):
    """
    -------------------------------------------------------------------------------------------------------
    ID: TRAM_02
    Name: Schedule Information for Trams
    Author: AlexT
    -------------------------------------------------------------------------------------------------------
    """

    def name(self) -> Text:
        return "action_find_next_tram"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = tracker.get_slot("transport_mode")

            logger.info(f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

            # Ensure the transport mode is 'tram'
            if not transport_mode or "tram" not in transport_mode.lower():
                dispatcher.utter_message(response="utter_invalid_mode")
                return []

            # Validate station names
            if not station_a or not station_b:
                dispatcher.utter_message(text="Please specify both the starting and destination stations for the tram.")
                return []

            # Use preloaded tram_stops to find stop IDs
            stop_a_id = GTFSUtils.get_station_id(station_a, tram_stops)
            stop_b_id = GTFSUtils.get_station_id(station_b, tram_stops)

            if not stop_a_id or not stop_b_id:
                dispatcher.utter_message(
                    text=f"Sorry, I couldn't find one or both of the stations: {station_a}, {station_b}."
                )
                return []

            # Get the current time for filtering
            current_time = datetime.now().strftime('%H:%M:%S')

            # Find the next tram trip using preloaded tram_stop_times
            if not isinstance(tram_stop_times.index, pd.MultiIndex):
                tram_stop_times.set_index(['stop_id', 'trip_id'], inplace=True, drop=False)

            trips_from_station_a = tram_stop_times.loc[stop_a_id].reset_index()
            trips_to_station_b = tram_stop_times.loc[stop_b_id].reset_index()

            # Filter for future trips and match them
            future_trips = trips_from_station_a[
                trips_from_station_a['departure_time'] >= current_time
            ]['trip_id'].unique()
            valid_trips = trips_to_station_b[
                trips_to_station_b['trip_id'].isin(future_trips)
            ]

            # Generate the response
            if not valid_trips.empty:
                next_trip = valid_trips.iloc[0]
                departure_time = GTFSUtils.parse_time(next_trip['departure_time'])

                # Convert parsed time to a user-friendly format
                if isinstance(departure_time, timedelta):
                    departure_time = (datetime.min + departure_time).strftime('%I:%M %p')

                response = f"The next tram from {station_a} to {station_b} departs at {departure_time}."
            else:
                response = f"Sorry, no upcoming trams were found from {station_a} to {station_b}."

            # Send the response
            dispatcher.utter_message(text=response)

        except Exception as e:
            # Handle exceptions and log the error
            logger.error(f"Failed to process 'action_find_next_tram': {str(e)}")
            dispatcher.utter_message(text="An unexpected error occurred while fetching the tram schedule. Please try again.")

        return []

class ActionCheckDisruptionsTrain(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: REQ_06
        Name: Real-time train updates
        Author: AlexT
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_disruptions_train"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Extract route name from user query
            query = tracker.latest_message.get('text')
            route_name = GTFSUtils.extract_route_name(query, routes_df)

            if not route_name:
                dispatcher.utter_message(text="I couldn't determine the train route. Please provide a valid route name (e.g., 'Frankston').")
                return []

            # Fetch disruptions for the route
            active_disruptions, route_id, error = GTFSUtils.check_route_and_fetch_disruptions(route_name, "train", routes_df)

            if error:
                dispatcher.utter_message(text=error)
                return []

            # Generate response based on disruptions
            if active_disruptions:
                table = [
                    [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
                    for d in active_disruptions
                ]
                response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
                dispatcher.utter_message(text=f"Active disruptions for the train {route_name} route:\n{response}")
            else:
                dispatcher.utter_message(text=f"There are no active train disruptions for the {route_name} route.")

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while checking disruptions. Please try again.")
            logger.error(f"Failed to check disruptions: {str(e)}")

        return []

class ActionCheckDisruptionsTram(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: TRAM_07
        Name: Real-time Tram updates
        Author: AlexT
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_disruptions_tram"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get("text").lower()  # Get the user's query
        print(f"Received query for tram: {query}")  # Debugging

        # Debug the dataset
        print("Loading tram routes dataset...")
        print(f"Available tram routes (short names): {tram_routes['route_short_name'].tolist()}")
        print(f"Available tram routes (long names): {tram_routes['route_long_name'].tolist()}")

        route_name = GTFSUtils.extract_route_name(query, tram_routes)
        if not route_name:
            dispatcher.utter_message(
                text="I couldn't determine the tram route. Please provide a valid route name (e.g., '109')."
            )
            return []

        # Fetch disruptions
        active_disruptions, route_id, error = GTFSUtils.check_route_and_fetch_disruptions(
            route_name, "tram", tram_routes
        )
        if error:
            dispatcher.utter_message(text=error)
            return []

        # Respond with disruptions
        if active_disruptions:
            table = [
                [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
                for d in active_disruptions
            ]
            response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
            dispatcher.utter_message(text=f"Active tram disruptions for Route {route_id}:\n{response}")
        else:
            dispatcher.utter_message(
                text=f"There are no active tram disruptions for Route {route_name} ({route_id})."
            )
        return []


class ActionCheckDisruptionsBus(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: BUS_07
        Name: Real-time BUS updates
        Author: AlexT
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_disruptions_bus"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get("text").lower()  # Get the user's query
        print(f"Received query for bus: {query}")  # Debugging

        # Debug the dataset
        print("Loading bus routes dataset...")
        print(f"Available bus routes (short names): {bus_routes['route_short_name'].tolist()}")
        print(f"Available bus routes (long names): {bus_routes['route_long_name'].tolist()}")

        route_name = GTFSUtils.extract_route_name(query, bus_routes)
        if not route_name:
            dispatcher.utter_message(
                text="I couldn't determine the bus route. Please provide a valid route name (e.g., '903')."
            )
            return []

        # Fetch disruptions
        active_disruptions, route_id, error = GTFSUtils.check_route_and_fetch_disruptions(
            route_name, "bus", bus_routes
        )
        if error:
            dispatcher.utter_message(text=error)
            return []

        # Respond with disruptions
        if active_disruptions:
            table = [
                [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
                for d in active_disruptions
            ]
            response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
            dispatcher.utter_message(text=f"Active bus disruptions for Route {route_id}:\n{response}")
        else:
            dispatcher.utter_message(
                text=f"There are no active bus disruptions for Route {route_name} ({route_id})."
            )
        return []

class ActionGenerateTrainMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_13
    	Name: Generate Map of Train Stations
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_generate_train_map"

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

            # Save the map to an HTML file
            map_filename = 'melbourne_train_stations_map.html'
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)  # Create the maps folder if it doesn't exist
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # Get the base URL from the environment variable
            server_base_url = os.getenv('SERVER_BASE_URL')

            # Fallback if the environment variable is not set
            if server_base_url is None:
                server_base_url = 'http://localhost:8080'  # Default value or fallback

            # Create and return the hyperlink using the base URL
            public_url = f"{server_base_url}/maps/{map_filename}"
            hyperlink = f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne train stations</a>"

            # Send the message to the user with the hyperlink
            dispatcher.utter_message(
                text=f"The map of Melbourne train stations has been generated. {hyperlink}")

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to generate map", e)
            raise

class ActionGenerateTramMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: TRAM_01
        Name: Tram Stations Map
        Author: AlexT
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_generate_tram_map"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Remove duplicates
            stops_map_df = tram_stops.drop_duplicates(subset=['stop_lat', 'stop_lon'])

            # Initialize map
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

            # Use MarkerCluster for better performance
            marker_cluster = MarkerCluster().add_to(melbourne_map)

            # Add markers with simplified popups
            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row['stop_lat'], row['stop_lon']],
                    popup=f"{row['stop_name']}",
                    tooltip=f"{row['stop_name']}"
                ).add_to(marker_cluster)

            # Save map to HTML
            map_filename = 'melbourne_tram_stops_map.html'
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # Send map link to user
            server_base_url = os.getenv('SERVER_BASE_URL', 'http://localhost:8080')
            public_url = f"{server_base_url}/maps/{map_filename}"
            hyperlink = f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne tram stops</a>"
            dispatcher.utter_message(text=f"The map of Melbourne tram stops has been generated. {hyperlink}")

        except Exception as e:
            logging.error(f"Error generating tram map: {e}")
            dispatcher.utter_message(text="An error occurred while generating the tram map.")
        return []

class ActionGenerateBusMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: BUS_01
        Name: Bus Stations Map
        Author: AlexT
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_generate_bus_map"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Ensure bus_stops has the required columns
            stops_map_df = bus_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

            # Add markers for bus stops
            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row['stop_lat'], row['stop_lon']],
                    popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
                    tooltip=row['stop_name']
                ).add_to(melbourne_map)

            # Save the map to an HTML file
            map_filename = 'melbourne_bus_stops_map.html'
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # Generate public URL for the map
            server_base_url = os.getenv('SERVER_BASE_URL', 'http://localhost:8080')
            public_url = f"{server_base_url}/maps/{map_filename}"
            hyperlink = f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne bus stops</a>"

            dispatcher.utter_message(text=f"The map of Melbourne bus stops has been generated. {hyperlink}")
        except Exception as e:
            logging.error(f"Error generating bus map: {e}")
            dispatcher.utter_message(text="An error occurred while generating the bus map.")
        return []

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
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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
                # Logic for one station
                trips_from_station = stop_times_df.loc[stop_a_id]
                trips_from_station = trips_from_station[trips_from_station['departure_time'] >= current_time]
                trips_from_station = trips_from_station.sort_values('departure_time').drop_duplicates(
                    subset=['departure_time']
                )

                if not trips_from_station.empty:
                    next_trips = trips_from_station[['departure_time']].head(5)
                    response = f"Upcoming train schedules from {station_a}:\n"
                    for idx, row in next_trips.iterrows():
                        departure_time = GTFSUtils.parse_time(row['departure_time'])
                        response += f"- Train at {(datetime.min + departure_time).strftime('%I:%M %p')}\n"
                else:
                    response = f"No upcoming trains found from {station_a}."
            else:
                # Logic for two stations
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
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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

            response = f"The best route from {station_a} to {station_b} is on the {route_name} towards {destination}.\n"
            response += f"The trip takes approximately {best_trip['travel_time'] / 60:.2f} minutes."

            # Create the route map given the trip id, including the transfers_df to highlight transfer stations
            hyperlink = GTFSUtils.generate_route_map(
                best_trip['trip_id'], station_a, station_b, stops_df, stop_times_df, dataset_path
            )
            if hyperlink:
                response += f"\n{hyperlink}"

            dispatcher.utter_message(text=response)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the best route", e)
            raise

class ActionCalculateTransfers(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_03 implementation
    	Name: Basic Route Planning
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
        "How many transfers are there between [Station A] and [Station B]?"
        -------------------------------------------------------------------------------------------------------
        Testing:
        Transfer Route: Ensure all transfer stations and the correct travel time are included.
        No Route Found: Confirm that the message correctly informs the user when no route is available. recommend the best route with transfer
        Invalid Stations: Check how the action handles cases where stations are not found or the travel time cannot be calculated.
    '''
    def name(self) -> Text:
        return "action_calculate_transfers"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

            # Extract stations from the user query
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(
                    text="Please specify both the starting and destination stations. Example: 'How many transfers between Richmond and Parliament?'"
                )
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]
            print(f"Extracted stations: {station_a}, {station_b}")

            # Calculate transfers and transfer stations
            transfers, transfer_stations = GTFSUtils.calculate_transfers(
                station_a, station_b, stops_df, stop_times_df
            )
            travel_time = GTFSUtils.calculate_route_travel_time(
                [station_a] + transfer_stations + [station_b], stops_df, stop_times_df
            )

            # Generate response based on the number of transfers
            if transfers == 0:
                response = (
                    f"There is a direct train from {station_a} to {station_b}, so no transfers are needed."
                )
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

            # Optional: Generate the route map hyperlink
            # best_trip_id = GTFSUtils.find_best_trip_id_with_transfers(
            #     station_a, station_b, transfer_stations, stop_times_df
            # )
            # if best_trip_id:
            #     hyperlink = GTFSUtils.generate_route_map(
            #         best_trip_id, station_a, station_b, stops_df, stop_times_df, dataset_path, transfers_df
            #     )
            #     if hyperlink:
            #         response += f"\n{hyperlink}"

            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while calculating transfers. Please try again.")
            GTFSUtils.handle_error(dispatcher, logger, "Failed to calculate transfers", e)

        return []

class ActionCheckDirectRoute(Action):
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
    def name(self) -> Text:
        return "action_check_direct_route"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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


class ActionFindBestRouteWithTransfers(Action):
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
    def name(self) -> Text:
        return "action_find_best_route_with_transfers"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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

class ActionCheckTrainChange(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_03 implementation
    	Name: Check Train Change
    	Author: AlexT
    	-------------------------------------------------------------------------------------------------------
    	"Do I need to change trains to get to [Station B] from [Station A]?"
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_train_change"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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

class ActionFindRouteWithLeastStops(Action):
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
    def name(self) -> Text:
        return "action_find_route_with_least_stops"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

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

class ValidateTransportMode(Action):
    """
    Custom action to validate the transport_mode slot.
    """

    VALID_TRANSPORT_MODES = ["train", "tram", "bus"]

    def name(self) -> str:
        return "validate_transport_mode"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:
        # Extract the value of the transport_mode slot
        transport_mode = tracker.get_slot("transport_mode")

        if not transport_mode:
            dispatcher.utter_message(
                text="It seems you haven't specified a mode of transport. Please choose train, tram, or bus."
            )
            return [SlotSet("transport_mode", None)]

        # Normalize the user input
        transport_mode = transport_mode.lower().strip()

        if transport_mode in self.VALID_TRANSPORT_MODES:
            dispatcher.utter_message(
                text=f"You have selected {transport_mode} as your mode of transport."
            )
            return [SlotSet("transport_mode", transport_mode)]
        else:
            dispatcher.utter_message(
                text=f"Sorry, '{transport_mode}' is not a valid mode of transport. Please choose train, tram, or bus."
            )
            return [SlotSet("transport_mode", None)]

''' -------------------------------------------------------------------------------------------------------
	
	Name: Directions and Mapping
	Author: LoganG
	-------------------------------------------------------------------------------------------------------
'''
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import subprocess
import sys
import os
import logging

logger = logging.getLogger(__name__)

class ActionRunDirectionScriptOriginal(Action):
    def name(self) -> Text:
        return "action_run_direction_script"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        location_from = tracker.get_slot('location_from')
        location_to = tracker.get_slot('location_to')
        
        if not location_from or not location_to:
            dispatcher.utter_message(text="Please provide both starting location and destination in the format: 'How do I get from [location] to [destination]'")
            return []

        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "userlocationmaps_executablepassingactions.py")
        
        try:
            result = subprocess.run([sys.executable, script_path, location_from, location_to], 
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

                    relative_path = os.path.relpath(map_file_path, current_dir)
                    map_url = f"http://localhost:8000/{relative_path.replace(os.sep, '/')}"

                    map_link = f"<a href='{map_url}' target='_blank'>Click here to view the route map</a>"
                    dispatcher.utter_message(text=f"I've generated a route map for you: {map_link}")
                else:
                    dispatcher.utter_message(text="The script returned incomplete output. Please try again.")
            else:
                dispatcher.utter_message(text="The direction script executed successfully, but no output was produced.")

        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip() if e.stderr else "Unknown error occurred during script execution."
            dispatcher.utter_message(text=f"An error occurred while running the script: {error_message}")
            logger.error(f"Script execution failed: {error_message}")
        except Exception as e:
            dispatcher.utter_message(text=f"An unexpected error occurred: {str(e)}")
            logger.error(f"Exception occurred: {str(e)}")

        return []

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
    
''' -------------------------------------------------------------------------------------------------------
	
	Name: Bus and Trains
	Author: LoganG
	-------------------------------------------------------------------------------------------------------
'''

import os
import sys
import time
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import logging

# Add the current actions folder to search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from multimodal_transit_router import MultimodalTransitRouter

logger = logging.getLogger(__name__)
class ActionRunDirectionScript(Action):
    def __init__(self):
        super().__init__()
        self.router = None
        logger = logging.getLogger(__name__)
        logger.info("ActionRunDirectionScript initialized")
    
    def name(self) -> Text:
        return "action_run_direction_script"
    
    def _initialize_router(self):
        """Initialize the router if not already initialized"""
        if self.router is None:
            try:
                init_start = time.time()
                logger.info("Starting router initialization...")
                
                dir_start = time.time()
                current_dir = os.path.dirname(os.path.abspath(__file__))
                os.chdir(current_dir)
                logger.info(f"Directory operations took {time.time() - dir_start:.2f} seconds")
                
                router_start = time.time()
                self.router = MultimodalTransitRouter()
                logger.info(f"Router creation took {time.time() - router_start:.2f} seconds")
                
                logger.info(f"Total initialization took {time.time() - init_start:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Failed to initialize transit router: {e}")
                raise
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            
        # Log debug information
        latest_message = tracker.latest_message
        logger.info("========== START DEBUG INFO ==========")
        logger.info(f"Latest message text: {latest_message.get('text')}")
        logger.info(f"Recognized intent: {latest_message.get('intent', {}).get('name')}")
        logger.info(f"Confidence score: {latest_message.get('intent', {}).get('confidence')}")
        logger.info(f"Extracted entities: {latest_message.get('entities', [])}")
        
        run_start = time.time()
        logger.info("Starting action execution")
        
        # Get locations from slots
        slots_start = time.time()
        location_from = tracker.get_slot('location_from')
        location_to = tracker.get_slot('location_to')
        logger.info(f"Final location_from: {location_from}")
        logger.info(f"Final location_to: {location_to}")
        logger.info(f"Slot extraction took {time.time() - slots_start:.2f} seconds")
        
        if not location_from or not location_to:
            error_msg = f"Missing {'origin' if not location_from else 'destination'} location"
            logger.error(f"Validation failed: {error_msg}")
            dispatcher.utter_message(text=f"Please provide both starting location and destination. {error_msg}.")
            return []

        try:
            self._initialize_router()
            
            # Get route
            route_start = time.time()
            directions, map_file = self.router.find_route(location_from, location_to)
            logger.info(f"Route finding took {time.time() - route_start:.2f} seconds")
            
            # Send response
            response_start = time.time()
            if isinstance(directions, str):
                # Send all directions in a single message
                dispatcher.utter_message(text=directions)
                
                # Handle map file if it exists
                if map_file and os.path.exists(map_file):
                    relative_path = os.path.relpath(map_file, current_dir)
                    map_url = f"http://localhost:8000/{relative_path.replace(os.sep, '/')}"
                    link_message = f'<a href="{map_url}" target="_blank">Click here to view the route map</a>'
                    dispatcher.utter_message(text=link_message, parse_mode="html")
            else:
                dispatcher.utter_message(text=directions)  # Error message
                
            logger.info(f"Response handling took {time.time() - response_start:.2f} seconds")
                
        except Exception as e:
            error_msg = f"An error occurred while finding the route: {str(e)}"
            logger.error(error_msg, exc_info=True)
            dispatcher.utter_message(text=error_msg)

        logger.info(f"Total action execution took {time.time() - run_start:.2f} seconds")
        return []


class ActionFindTransferTramRoute(Action):

    '''
    ----------------------------------------------------------------------
    tram routing with transfers - unfinished
    by: JubalK
    -----------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_tram_route_with_transfers"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')

            print(f"tram stops dataframe: {tram_stops.head()}")
            print(f"tram timings dataframe: {tram_stop_times.head()}")
            print(f"tram timings columns: {tram_stop_times.columns}")
            print(f"tram stops dataframe: {tram_stops.columns}")

            query_normalized = query.lower().strip()
            logger.info(f"Normalized query: {query_normalized}")

            print(f"DataFrame passed to extract_stations_from_query: {bus_stops.head()['normalized_stop_name']}")
            # Extract stations from query
            extracted_stations = GTFSUtils.extract_stations_from_query(query, tram_stops)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1]

            if not station_a or not station_b:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, tram_stops)
            stop_b_id = GTFSUtils.get_station_id(station_b, tram_stops)

            print(f"stop id for stop a: {stop_a_id}")
            print(f"stop id for stop a: {stop_b_id}")

            print(f"Station A: {station_a}, ID: {stop_a_id}")
            print(f"Station B: {station_b}, ID: {stop_b_id}")

            if stop_a_id not in tram_stop_times.index.get_level_values('stop_id'):
                dispatcher.utter_message(text=f"Stop ID for {station_a} not found.")
                return []
            if stop_b_id not in tram_stop_times.index.get_level_values('stop_id'):
                dispatcher.utter_message(text=f"Stop ID for {station_b} not found.")
                return []

            # Find trips for starting and destination stops
            stop_a_trips = tram_stop_times.loc[stop_a_id].reset_index()
            stop_b_trips = tram_stop_times.loc[stop_b_id].reset_index()

            print(f"stop_a_trips: {stop_a_trips}")
            print(f"stop_b_trips: {stop_b_trips}")

            # Join trips to find possible transfer points
            transfer_candidates = pd.merge(
                stop_a_trips, tram_stop_times.reset_index(), on="trip_id", suffixes=('_a', '_transfer')
            )

            print(f"transfer_candidates: {transfer_candidates}")

            # Filter valid transfer points where stop sequence aligns
            transfer_candidates = transfer_candidates[
                transfer_candidates['stop_sequence_a'] < transfer_candidates['stop_sequence_transfer']
                ]

            transfer_candidates = transfer_candidates.rename(columns={
                "stop_id_transfer": "transfer_stop_id",
                "arrival_time_transfer": "arrival_time_transfer"
            })

            print(f"transfer_candidates after rename: {transfer_candidates}")

            # Identify transfer stops and destination trips
            potential_transfers = pd.merge(
                transfer_candidates, stop_b_trips, left_on="transfer_stop_id", right_on="stop_id",
                suffixes=('_transfer', '_b')
            )

            # Filter valid transfers (sequence constraints)
            potential_transfers = potential_transfers[
                potential_transfers['stop_sequence_transfer'] < potential_transfers['stop_sequence_b']
                ]

            if potential_transfers.empty:
                dispatcher.utter_message(text="No transfer route found between the two stations.")
                return []

            # Calculate travel times for valid transfers
            potential_transfers['travel_time'] = (
                    potential_transfers['arrival_time_b'].apply(GTFSUtils.parse_time) -
                    potential_transfers['arrival_time_a'].apply(GTFSUtils.parse_time)
            ).dt.total_seconds()

            # Find the best transfer option
            best_transfer = potential_transfers.loc[potential_transfers['travel_time'].idxmin()]

            transfer_station = tram_stops.loc[best_transfer['transfer_stop_id'], 'stop_name']
            first_route_name = tram_routes.loc[
                tram_trips.loc[tram_trips['trip_id'] == best_transfer['trip_id_a'], 'route_id'].values[0],
                'route_long_name'
            ]
            second_route_name = tram_routes.loc[
                tram_trips.loc[tram_trips['trip_id'] == best_transfer['trip_id_b'], 'route_id'].values[0],
                'route_long_name'
            ]
            destination = tram_trips.loc[tram_trips['trip_id'] == best_transfer['trip_id_b'], 'trip_headsign'].values[0]

            response = (f"The best transfer route from {station_a} to {station_b} is as follows:\n"
                        f"1. Take the {first_route_name} to {transfer_station}.\n"
                        f"2. Transfer at {transfer_station} to the {second_route_name} towards {destination}.\n"
                        f"The trip will take approximately {best_transfer['travel_time'] / 60:.2f} minutes.")

            dispatcher.utter_message(text=response)

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find a transfer route", e)


class ActionFindTramRoute(Action):
    '''
    ----------------------------------------------------------------------
    tram routing (direct connections)
    by: JubalK
    -----------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_direct_tram_route"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')

            print(f"tram stops dataframe: {tram_stops.head()}")
            print(f"tram timings dataframe: {tram_stop_times.head()}")
            print(f"tram timings columns: {tram_stop_times.columns}")
            print(f"tram stops dataframe: {tram_stops.columns}")

            query_normalized = query.lower().strip()
            logger.info(f"Normalized query: {query_normalized}")

            print(f"DataFrame passed to extract_stations_from_query: {tram_stops.head()['normalized_stop_name']}")
            extracted_stations = GTFSUtils.extract_stations_from_query(query, tram_stops)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or (not station_b and "to" in query.lower()):
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils. get_station_id(station_a, tram_stops)
            stop_b_id = GTFSUtils. get_station_id(station_b, tram_stops) if station_b else None

            print(f"stop id for stop a: {stop_a_id}")
            print(f"stop id for stop a: {stop_b_id}")

            print(f"Station A: {station_a}, ID: {stop_a_id}")
            print(f"Station B: {station_b}, ID: {stop_b_id}")

            if stop_a_id not in tram_stop_times.index.get_level_values('stop_id'):
                dispatcher.utter_message(text=f"Stop ID for {station_a} not found.")
                return []
            if stop_b_id not in tram_stop_times.index.get_level_values('stop_id'):
                dispatcher.utter_message(text=f"Stop ID for {station_b} not found.")
                return []

            if not station_a or not station_b:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            print(station_a)
            print(station_b)

            stop_a_times = tram_stop_times.loc[stop_a_id][['stop_sequence', 'arrival_time']].reset_index()
            stop_b_times = tram_stop_times.loc[stop_b_id][['stop_sequence', 'arrival_time']].reset_index()

            print(f"stop_a_id for stop a: {stop_a_times}")
            print(f"stop_b_id for stop a: {stop_b_times}")

            common_trip_ids = set(stop_a_times['trip_id']).intersection(set(stop_b_times['trip_id']))
            print(f"Common trip IDs: {common_trip_ids}")

            merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))
            print(f"Merged stop times:\n{merged.head()}")

            valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()
            print(f"valid trips:\n{valid_trips.head()}")


            if valid_trips.empty:
                dispatcher.utter_message(text="No direct route found between the two stations.")
                return []

            valid_trips['arrival_time_a'] = valid_trips['arrival_time_a'].apply(GTFSUtils.parse_time)
            valid_trips['arrival_time_b'] = valid_trips['arrival_time_b'].apply(GTFSUtils.parse_time)
            valid_trips['travel_time'] = (
                    valid_trips['arrival_time_b'] - valid_trips['arrival_time_a']).dt.total_seconds()

            print(f"valid trips columns:\n{valid_trips.columns}")
            print(f"valid trips:\n{valid_trips.head()}")

            best_trip = valid_trips.loc[valid_trips['travel_time'].idxmin()]
            print(f"best trips:\n{best_trip}")

            route_id = tram_trips.loc[tram_trips['trip_id'] == best_trip['trip_id'], 'route_id'].values[0]
            route_name = tram_routes.loc[tram_routes['route_id'] == route_id, 'route_long_name'].values[0]
            destination = tram_trips.loc[tram_trips['trip_id'] == best_trip['trip_id'], 'trip_headsign'].values[0]
            print(f"route ID:\n{route_id}")
            print(f"route name:\n{route_name}")
            print(f"destination:\n{destination}")

            response = f"The best route from {station_a} to {station_b} is on the {route_name} towards {destination} \n The trip taking approximately {best_trip['travel_time'] / 60:.2f} minutes."

            print(response)

            hyperlink = GTFSUtils.generate_route_map(best_trip['trip_id'], station_a, station_b, tram_stops, tram_stop_times,
                                                     dataset_path)
            if hyperlink:
                response += f"\n{hyperlink}"

            dispatcher.utter_message(text=response)

        except Exception as e: GTFSUtils.handle_error(dispatcher, logger, "Failed to find the best route", e)




# Ross Start Actions
class ActionFindNearestStation(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: TRAIN
        Name: Find Nearest Station
        Author: RossP
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_find_nearest_station"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message.get("entities", [])

        unique_val = []
        seen_val = set()
        
        for entity in entities:
            if entity['value'] not in seen_val:
                unique_val.append(entity['value'])
                seen_val.add(entity['value']) 
            if len(unique_val) == 2:
                break   
         
        address_entity = ", ".join(unique_val) 
        logger.debug(address_entity)
        
        #get lat and long of location form google API
        addll = GTFSUtils.getAddressLatLong(address = address_entity)
        logger.debug(addll)

        #check distance to all stations
        closStat = GTFSUtils.checkDistancetoAllStation(addll['latitude'],addll['longitude'])
                        
        
        if address_entity:
                dispatcher.utter_message(text = f"The closest station to {address_entity} is {closStat['closest_station_name']}")
        else: 
            dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
        return []

class ActionFindNearestTramStop(Action):
    ''' -------------------------------------------------------------------------------------------------------
    ID: TRAM
    Name: Find Nearest Tram Stop
    Author: RossP
    -------------------------------------------------------------------------------------------------------
    ''' 
    def name(self) -> Text:
        return "action_find_nearest_tram_stop"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message.get("entities", [])

        unique_val = []
        seen_val = set()
        
        for entity in entities:
            if entity['value'] not in seen_val:
                unique_val.append(entity['value'])
                seen_val.add(entity['value']) 
            if len(unique_val) == 2:
                break   
         
        address_entity = ", ".join(unique_val) 
        logger.debug(address_entity)
        
        #get lat and long of location form google API
        addll = GTFSUtils.getAddressLatLong(address = address_entity)
        logger.debug(addll)

        #check distance to all stations
        closStat = GTFSUtils.checkDistancetoAllTramsStops(addll['latitude'],addll['longitude'])
                        
        
        if address_entity:
                dispatcher.utter_message(text = f"The closest Tram Stop to {address_entity} is {closStat['closest_station_name']}")
        else: 
            dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
        return []

class ActionFindNearestBusStop(Action):    
    ''' -------------------------------------------------------------------------------------------------------
    ID: BUS
    Name: Find Nearest Bus Stop
    Author: RossP
    -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_nearest_bus_stop"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message.get("entities", [])

        unique_val = []
        seen_val = set()
        
        for entity in entities:
            if entity['value'] not in seen_val:
                unique_val.append(entity['value'])
                seen_val.add(entity['value']) 
            if len(unique_val) == 2:
                break   
         
        address_entity = ", ".join(unique_val) 
        logger.debug(address_entity)
        
        #get lat and long of location form google API
        addll = GTFSUtils.getAddressLatLong(address = address_entity)
        logger.debug(addll)

        #check distance to all stations
        closStat = GTFSUtils.checkDistancetoAllBusStops(addll['latitude'],addll['longitude'])
                        
        
        if address_entity:
                dispatcher.utter_message(text = f"The closest Bus Stop to {address_entity} is {closStat['closest_station_name']}")
        else: 
            dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
        return []
    
class ActionMapTransportInArea(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: Mulit_01
        Name: Map transport in area
        Author: RossP
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_map_transport_in_area"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = tracker.latest_message.get("entities", [])
        print(entities)
        # parse entities for transport mode and location data
        for entity in entities:
            if entity['entity'] == 'transport_mode_rp':
                transport_mode = entity['value']
            
            if entity['entity'] == 'locationa':
                location = entity['value']
        print(transport_mode,  location)
        
        #get lat and long of location form google API
        locll = GTFSUtils.getAddressLatLong(address = location)
        logger.debug(locll)

        #get list of lat and long of stations around 1k from location, with transport mode
        lstns = GTFSUtils.getListOfStationsWithin1k(locll, transport_mode)
        print(lstns)        
        
        try: 
            # Initialize map
            melbourne_map = folium.Map(location=[locll['latitude'], locll['longitude']],
                                       zoom_start=15)

            # Use MarkerCluster for better performance
            marker_cluster = MarkerCluster().add_to(melbourne_map)

            # Add markers with simplified popups
            for stns in lstns:
                folium.Marker(
                    location=[stns['lat'], stns['lon']],
                    popup=f"{stns['stop']}",
                    tooltip=f"{stns['stop']}"
                ).add_to(marker_cluster)

            # Save map to HTML
            map_filename = 'stops_map.html'
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # Send map link to user
            server_base_url = os.getenv('SERVER_BASE_URL', 'http://localhost:8080')
            public_url = f"{server_base_url}/maps/{map_filename}"
            hyperlink = f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne {transport_mode} stops</a>"
            dispatcher.utter_message(text=f"A map of {transport_mode} stops within 900 meters of {location} has been generated. {hyperlink}")

        except Exception as e:
            logging.error(f"Error generating tram map: {e}")
            dispatcher.utter_message(text="An error occurred while generating the tram map.")
        return []

# Ross Finish Actions