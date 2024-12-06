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
from folium.plugins import MarkerCluster
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

class ActionFindNextBus(Action):
    ''' -------------------------------------------------------------------------------------------------------
         ID: BUS_02
         Name: Schedule Information for Buses
         Author: AlexT
         -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_next_bus"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, bus_stops)

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or (not station_b and "to" in query.lower()):
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, bus_stops)
            stop_b_id = GTFSUtils.get_station_id(station_b, bus_stops) if station_b else None

            current_time = datetime.now().strftime('%H:%M:%S')

            if not isinstance(bus_stop_times.index, pd.MultiIndex):
                bus_stop_times.set_index(['stop_id', 'trip_id'], inplace=True, drop=False)

            if not station_b:
                trips_from_station = bus_stop_times.loc[stop_a_id]
                trips_from_station = trips_from_station[trips_from_station['departure_time'] >= current_time]
                trips_from_station = trips_from_station.sort_values('departure_time').drop_duplicates(
                    subset=['departure_time'])

                if not trips_from_station.empty:
                    next_trips = trips_from_station[['departure_time']].head(5)
                    response = f"Upcoming bus schedules from {station_a}:\n"
                    for idx, row in next_trips.iterrows():
                        departure_time = GTFSUtils.parse_time(row['departure_time'])
                        response += f"- Bus at {(datetime.min + departure_time).strftime('%I:%M %p')}\n"
                else:
                    response = f"No upcoming buses found from {station_a}."
            else:
                trips_from_station_a = bus_stop_times.loc[stop_a_id].reset_index()
                trips_to_station_b = bus_stop_times.loc[stop_b_id].reset_index()

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
                    response = f"The next bus from {station_a} to {station_b} leaves at {next_trip_time}."
                else:
                    response = f"No upcoming buses found from {station_a} to {station_b}."

            dispatcher.utter_message(text=response)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the next bus", e)
            raise

class ActionFindNextTram(Action):
    ''' -------------------------------------------------------------------------------------------------------
         ID: TRAM_02
         Name: Schedule Information for Trams        
         Author: AlexT
         -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_next_tram"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            query = tracker.latest_message.get('text')
            extracted_stations = GTFSUtils.extract_stations_from_query(query, tram_stops)

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or (not station_b and "to" in query.lower()):
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = GTFSUtils.get_station_id(station_a, tram_stops)
            stop_b_id = GTFSUtils.get_station_id(station_b, tram_stops) if station_b else None

            current_time = datetime.now().strftime('%H:%M:%S')

            if not isinstance(tram_stop_times.index, pd.MultiIndex):
                tram_stop_times.set_index(['stop_id', 'trip_id'], inplace=True, drop=False)

            if not station_b:
                trips_from_station = tram_stop_times.loc[stop_a_id]
                trips_from_station = trips_from_station[trips_from_station['departure_time'] >= current_time]
                trips_from_station = trips_from_station.sort_values('departure_time').drop_duplicates(
                    subset=['departure_time'])

                if not trips_from_station.empty:
                    next_trips = trips_from_station[['departure_time']].head(5)
                    response = f"Upcoming tram schedules from {station_a}:\n"
                    for idx, row in next_trips.iterrows():
                        departure_time = GTFSUtils.parse_time(row['departure_time'])
                        response += f"- Tram at {(datetime.min + departure_time).strftime('%I:%M %p')}\n"
                else:
                    response = f"No upcoming trams found from {station_a}."
            else:
                trips_from_station_a = tram_stop_times.loc[stop_a_id].reset_index()
                trips_to_station_b = tram_stop_times.loc[stop_b_id].reset_index()

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
                    response = f"The next tram from {station_a} to {station_b} leaves at {next_trip_time}."
                else:
                    response = f"No upcoming trams found from {station_a} to {station_b}."

            dispatcher.utter_message(text=response)
        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the next tram", e)
            raise

class ActionGenerateMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
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

''' -------------------------------------------------------------------------------------------------------	
	Name: Generate route map
	Author: AlexT	
    -------------------------------------------------------------------------------------------------------                
'''
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