import spacy
import folium
import webbrowser
from folium.plugins import MarkerCluster
import os
import re
from io import BytesIO
import zipfile
import requests
import pandas as pd
import logging
from typing import Any, Text, Dict, List, Optional
from fuzzywuzzy import process, fuzz
from datetime import datetime, timedelta
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from typing import Tuple
from collections import deque
import certifi
from actions.gtfs_utils import GTFSUtils
from sanic import Sanic
from sanic.response import text
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from rasa_sdk.types import DomainDict
from dotenv import load_dotenv
from difflib import get_close_matches
#from actions.traffic_route 
import hashlib
import hmac
import urllib.parse
from tabulate import tabulate
from pathlib import Path
from rasa_sdk.events import SlotSet, EventType, AllSlotsReset
from .tomtom_utils import tt_geocode, tt_route, fmt_time, fmt_km

# This is to skip the favicon
app = Sanic("custom_action_server")
@app.route("/favicon.ico")
async def favicon(request):
    return text("")  # Ignore the favicon request

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='rasa.shared.utils.io')

load_dotenv()
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
# Juveria- start global variable-------------------------
station_data['norm_name'] = (
    station_data['Station Name']
      .str.replace(' railway station', '', regex=False)
      .str.replace(' station', '', regex=False)
      .str.replace(' railway', '', regex=False)
      .str.replace('[()/_-]', ' ', regex=True)
      .str.replace(r'\s+', ' ', regex=True)
      .str.strip()
)
# Juveria-End GLobal variable-------------------
load_dotenv(dotenv_path="./key.env")
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.warning(msg="Your google api key is not set, please set it inside key.env file")

class ActionFindNextTram(Action):
    """
    -------------------------------------------------------------------------------------------------------
    ID: TRAM_02
    Name: Schedule Information for Trams
    Author: AlexT
    Modified: Juveria Nishath, +multi-platform fix
    -------------------------------------------------------------------------------------------------------
    """

    def name(self) -> Text:
        return "action_find_next_tram"

    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Treat this action as 'tram' regardless of transport_mode
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            user_text = (tracker.latest_message or {}).get("text", "") or ""
            # --- MODE GUARD: if the user clearly asked for a *train*, hand over ---
            if "train" in user_text.lower():
                return ActionFindNextTrain().run(dispatcher, tracker, domain)
# ----------------------------------------------------------------------

            # Fallback: extract from user text when slots are empty
            if not station_a or not station_b:
                extracted = GTFSUtils.extract_stations_from_query(user_text, tram_stops)
                if extracted:
                    station_a = station_a or extracted.get("station_a")
                    station_b = station_b or extracted.get("station_b")

            if not station_a or not station_b:
                dispatcher.utter_message(
                    "Please tell me both tram stops, e.g. “next tram from A to B”.")
                return []

            # Resolve a primary id (sanity/logs only)
            stop_a_id = GTFSUtils.get_stop_id(station_a, tram_stops)
            stop_b_id = GTFSUtils.get_stop_id(station_b, tram_stops)
            if not stop_a_id or not stop_b_id:
                dispatcher.utter_message(
                    "Sorry, I couldn't match those stops. Try the exact stop name shown on the green sign (e.g. “La Trobe St/Swanston St #5”)."
                )
                return []

            # Ensure index shape (define st FIRST!)
            st = tram_stop_times
            if not isinstance(st.index, pd.MultiIndex):
                st = st.set_index(["stop_id", "trip_id"], drop=False)

            # Build candidate stop_id sets for both stops (handles multiple platforms/directions)
            a_ids = GTFSUtils.collect_platform_ids(station_a, tram_stops, st)
            b_ids = GTFSUtils.collect_platform_ids(station_b, tram_stops, st)

            logger.info(f"[TRAM] A='{station_a}' candidates={a_ids}")
            logger.info(f"[TRAM] B='{station_b}' candidates={b_ids}")

            if not a_ids or not b_ids:
                dispatcher.utter_message("I can't find one of those stops in the tram GTFS.")
                return []

            now_str = datetime.now().strftime("%H:%M:%S")

            def earliest_candidate(a_list, b_list):
                best = None
                best_pair = None
                idx0 = st.index.get_level_values(0)

                for a_id in a_list:
                    if a_id not in idx0:
                        continue
                    a_times = st.loc[a_id].reset_index()
                    if isinstance(a_times, pd.Series):
                        a_times = a_times.to_frame().T
                    a_times = a_times[a_times["departure_time"] >= now_str]
                    if a_times.empty:
                        continue

                    for b_id in b_list:
                        if b_id not in idx0:
                            continue
                        b_times = st.loc[b_id].reset_index()
                        if isinstance(b_times, pd.Series):
                            b_times = b_times.to_frame().T

                        merged = a_times.merge(b_times, on="trip_id", suffixes=("_a", "_b"))
                        merged = merged[merged["stop_sequence_a"] < merged["stop_sequence_b"]]
                        if merged.empty:
                            continue

                        cand = merged.sort_values("departure_time_a").iloc[0]
                        if best is None or cand["departure_time_a"] < best["departure_time_a"]:
                            best = cand
                            best_pair = (a_id, b_id)
                return best, best_pair

            # Try normal direction
            best, pair = earliest_candidate(a_ids, b_ids)

            # If none, try auto-swap once (user may have reversed stops)
            swapped_used = False
            if best is None:
                best, pair = earliest_candidate(b_ids, a_ids)
                swapped_used = best is not None

            if best is None:
                dispatcher.utter_message(
                    "No upcoming direct trams found between those stops. They may be on different lines or direction. "
                    "Try reversing the stops or ask for a transfer route."
                )
                return []

        # Times
            depart_dt = GTFSUtils.parse_time(best["departure_time_a"])
            arrive_dt = GTFSUtils.parse_time(best["arrival_time_b"])
            depart = (datetime.min + depart_dt).strftime("%I:%M %p")
            arrive = (datetime.min + arrive_dt).strftime("%I:%M %p")

            # --- Route number + headsign lookup ---
            route_no = route_name = headsign = None
            try:
                trip_id = best["trip_id"]
                trip_row = tram_trips.loc[tram_trips["trip_id"] == trip_id]
                route_id = trip_row["route_id"].iloc[0] if not trip_row.empty else None
                headsign = trip_row["trip_headsign"].iloc[0] if not trip_row.empty else None

                if route_id is not None:
                    route_row = tram_routes.loc[tram_routes["route_id"] == route_id]
                    if not route_row.empty:
                        route_no = route_row["route_short_name"].iloc[0]
                        route_name = route_row["route_long_name"].iloc[0]
            except Exception:
                pass

            suffix = ""
            if route_no or headsign:
                parts = []
                if route_no:
                    parts.append(f"Route {route_no}")
                if headsign:
                    parts.append(f"to {headsign}")
                suffix = " (" + " ".join(parts) + ")"

            # Message
            if swapped_used:
                dispatcher.utter_message(
                    f"It looks like you meant the opposite direction.\n"
                    f"The next tram from {station_b} to {station_a} leaves at {depart} and arrives around {arrive}{suffix}."
                )
            else:
                dispatcher.utter_message(
                    f"The next tram from {station_a} to {station_b} leaves at {depart} and arrives around {arrive}{suffix}."
                )

            logger.info(
                f"[TRAM] Chosen stop_ids pair={pair}, trip_id={best['trip_id']}, "
                f"depart={best['departure_time_a']}, arrive={best['arrival_time_b']}, route_no={route_no}, headsign={headsign}"
            )
            return []

        except Exception as e:
            logger.exception("Error in ActionFindNextTram", exc_info=True)
            dispatcher.utter_message("Sorry — something went wrong while fetching tram times.")
            return []

# class ActionCheckDisruptionsTrain(Action):
#     ''' -------------------------------------------------------------------------------------------------------
#         ID: REQ_06
#         Name: Real-time train updates
#         Author: AlexT
#         -------------------------------------------------------------------------------------------------------
#     '''
#     def name(self) -> Text:
#         return "action_check_disruptions_train"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         try:
#             # Extract route name from user query
#             query = tracker.latest_message.get('text')
#             station_name = tracker.get_slot('route_name')
#             #route_name = GTFSUtils.extract_route_name(query, routes_df)
#             route_name = GTFSUtils.extract_route_name(station_name, routes_df)

#             if not route_name:
#                 dispatcher.utter_message(text="I couldn't determine the train route. Please provide a valid route name (e.g., 'Frankston').")
#                 return []

#             # Fetch disruptions for the route
#             active_disruptions, route_id, error = GTFSUtils.fetch_disruptions_by_route(route_name, "train", routes_df)
#             if error:
#                 dispatcher.utter_message(text=error)
#                 return []

#             # Generate response based on disruptions
#             if active_disruptions:
#                 table = [
#                     [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
#                     for d in active_disruptions
#                 ]
#                 response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
#                 dispatcher.utter_message(text=f"Active disruptions for the train {route_name} route:\n{response}")
#             else:
#                 dispatcher.utter_message(text=f"There are no active train disruptions for the {route_name} route.")

#         except Exception as e:
#             dispatcher.utter_message(text="An error occurred while checking disruptions. Please try again.")
#             logger.error(f"Failed to check disruptions: {str(e)}")

#         return []

# class ActionCheckDisruptionsTram(Action):
#     ''' -------------------------------------------------------------------------------------------------------
#         ID: TRAM_07
#         Name: Real-time Tram updates
#         Author: AlexT
#         -------------------------------------------------------------------------------------------------------
#     '''
#     def name(self) -> Text:
#         return "action_check_disruptions_tram"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         query = tracker.latest_message.get("text").lower()  # Get the user's query
#         print(f"Received query for tram: {query}")  # Debugging

#         tram_number = tracker.get_slot("route_number")

#         # Debug the dataset
#         print("Loading tram routes dataset...")

#         #route_name = GTFSUtils.extract_route_name(query, tram_routes)
#         route_name = GTFSUtils.extract_route_name(tram_number, tram_routes)
#         if not route_name:
#             dispatcher.utter_message(
#                 text="I couldn't determine the tram route. Please provide a valid route name (e.g., '109')."
#             )
#             return []

#         # Fetch disruptions
#         active_disruptions, route_id, error = GTFSUtils.fetch_disruptions_by_route(
#             route_name, "tram", tram_routes
#         )
#         if error:
#             dispatcher.utter_message(text=error)
#             return []

#         # Respond with disruptions
#         if active_disruptions:
#             table = [
#                 [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
#                 for d in active_disruptions
#             ]
#             response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
#             dispatcher.utter_message(text=f"Active tram disruptions for Route {route_id}:\n{response}")
#         else:
#             dispatcher.utter_message(
#                 text=f"There are no active tram disruptions for Route {route_name} ({route_id})."
#             )
#         return []


# class ActionCheckDisruptionsBus(Action):
#     ''' -------------------------------------------------------------------------------------------------------
#         ID: BUS_07
#         Name: Real-time BUS updates
#         Author: AlexT
#         -------------------------------------------------------------------------------------------------------
#     '''
#     def name(self) -> Text:
#         return "action_check_disruptions_bus"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         query = tracker.latest_message.get("text").lower()  # Get the user's query
#         print(f"Received query for bus: {query}")  # Debugging

#         bus_number = tracker.get_slot("route_number")

#         # Debug the dataset
#         print("Loading bus routes dataset...")


#         # route_name = GTFSUtils.extract_route_name(query, bus_routes)
#         route_name = GTFSUtils.extract_route_name(bus_number, bus_routes)
#         if not route_name:
#             dispatcher.utter_message(
#                 text="I couldn't determine the bus route. Please provide a valid route name (e.g., '903')."
#             )
#             return []

#         # Fetch disruptions
#         active_disruptions, route_id, error = GTFSUtils.fetch_disruptions_by_route(
#             route_name, "bus", bus_routes
#         )
#         if error:
#             dispatcher.utter_message(text=error)
#             return []

#         # Respond with disruptions
#         if active_disruptions:
#             table = [
#                 [d["disruption_id"], d["title"], d["disruption_type"], d["from_date"], d["to_date"]]
#                 for d in active_disruptions
#             ]
#             response = tabulate(table, headers=["ID", "Title", "Type", "Start", "End"], tablefmt="pretty")
#             dispatcher.utter_message(text=f"Active bus disruptions for Route {route_id}:\n{response}")
#         else:
#             dispatcher.utter_message(
#                 text=f"There are no active bus disruptions for Route {route_name} ({route_id})."
#             )
#         return []

## Andre Start Action ==========================

class ActionResetAllSlots(Action):
    def name(self):
        return "action_reset_all_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]):
        return [AllSlotsReset()]

class ActionCheckDisruptions(Action):
    ''' -------------------------------------------------------------------------------------------------------
        Name: Real-time disruption updates
        Author: AlexT
        Modifier: Andre Nguyen
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_check_disruptions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Extract route name from user query
            query = tracker.latest_message.get('text')
            route_name = tracker.get_slot('route_name')
            transport_mode = tracker.get_slot('transport_mode')
            #route_name = GTFSUtils.extract_route_name(query, routes_df)
            routes_dataframe = routes_df
            if transport_mode.lower() == "bus":
                routes_dataframe = bus_routes
            elif transport_mode.lower() == "tram":
                routes_dataframe = tram_routes

            route_name = GTFSUtils.extract_route_name(route_name, routes_dataframe)

            if not route_name:
                dispatcher.utter_message(text="I couldn't determine the train route. Please provide a valid route name (e.g., 'Frankston').")
                return []

            # Fetch disruptions for the route
            active_disruptions, route_id, error = GTFSUtils.fetch_disruptions_by_route(route_name, transport_mode, routes_dataframe)
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
                dispatcher.utter_message(text=f"There are no active {transport_mode} disruptions for the {route_name} route.")

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while checking disruptions. Please try again.")
            logger.error(f"Failed to check disruptions: {str(e)}")

        return [SlotSet("transport_mode", None)]

ALLOWED_PUBLIC_TRANSPORT = ["train", "tram", "bus"]
class ValidateNearestTransportForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_nearest_transport_form"

    def validate_transport_mode(
        self,
        value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if value and value.lower() in ALLOWED_PUBLIC_TRANSPORT:
            return {"transport_mode": value.lower()}
        dispatcher.utter_message(text="Please specify a valid transport mode (train, tram, or bus).")
        return {"transport_mode": None}

    def validate_address(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        if value and isinstance(value, str):
            doc = nlp(value)
            potential_address = [ent.text for ent in doc.ents]
            
            if len(potential_address) > 0:
                dispatcher.utter_message(text=f"your address is: {value}")
                user_lat, user_lon = geocode_address(value)
                return {"address": f"{user_lat},{user_lon}"}
        dispatcher.utter_message(text="Please provide a valid address.")
        return {"address": None}

class ActionFindNearestPublicTransport(Action):
    ''' -------------------------------------------------------------------------------------------------------
        Name: Store user's address
        Author: Andre Nguyen
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_submit_nearest_transport"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            address = tracker.get_slot("address")
            transport_mode = tracker.get_slot("transport_mode")
 
            address_split = address.split(',')
            user_lat, user_lon = address_split[0], address_split[1]
            stops_data = stops_df
            routes_data = routes_df
            if transport_mode == "tram":
                stops_data = tram_stops
                routes_data = tram_routes
            elif transport_mode == "bus":
                stops_data = bus_stops
                routes_data = bus_routes

            # Calculate distance to each stop
            stops_data['distance'] = stops_data.apply(
                lambda row: geodesic((user_lat, user_lon), (row['stop_lat'], row['stop_lon'])).km,
                axis=1
            )
            nearby_stops, message = GTFSUtils.find_all_nearby_stops(address, transport_mode, stops_data)
            if not nearby_stops.empty:
                table_data = nearby_stops[['stop_name', 'wheelchair_boarding', 'distance', 'num_of_disruption']].copy().head(10)
                table_data["wheelchair_boarding"] = table_data["wheelchair_boarding"].astype(str)
                table_data['wheelchair_boarding'] = table_data['wheelchair_boarding'].apply(
                    lambda x: 'Yes' if x == "1.0" else 'No'
                )

                # Format table using tabulate
                table = tabulate(
                    table_data,
                    headers=['Name', 'Wheelchair Boarding', 'Distance', 'Number of disruption'],
                    tablefmt='pretty',
                    floatfmt='.2f',
                    showindex=False
                )

                # Send response
                dispatcher.utter_message(text="Ensure to check disruption of your prefered station!")
                message = f"{transport_mode.upper()} Stops within 20 km of {address} for :\n\n{table}"
            
            dispatcher.utter_message(text=message)

            return []

        except Exception as e:
            dispatcher.utter_message(text=f"Error occurred in finding {transport_mode} near the address: {address}. Please try again.")
            logger.error(f"Failed to find nearest public transport: {transport_mode}: {str(e)}")

        return []

class ActionAskForPublicTransportMode(Action):
    ''' -------------------------------------------------------------------------------------------------------
        Name: Find nearest public transport
        Author: Andre Nguyen
        -------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_ask_transport_mode"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict) -> List[EventType]:
        transport_mode = ["Train", "Tram", "Bus"]
        dispatcher.utter_message(
            text="Please choose your mode of public transport: [Train, Tram or Bus]",
            buttons=[{"title": mode, "payload": mode} for mode in transport_mode],
        )

        return []


## Andre End Action
class ActionGenerateTrainMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_13
    	Name: Generate Map of Train Stations
    	Author: AlexT
        Modifier: Juveria Nishath
    	-------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_generate_train_map"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        try:
            # Build the train stops map
            stops_map_df = stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]]
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)

            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
                    tooltip=row["stop_name"],
                ).add_to(melbourne_map)

            # Save the map
            map_filename = "melbourne_train_stations_map.html"
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # Build an HTTP link (served by: python -m http.server 8000)
            server_base_url = os.getenv("MAP_SERVER_BASE_URL", "http://localhost:8000")
            public_url = f"{server_base_url}/maps/{map_filename}"

            # Single message with one link only
            dispatcher.utter_message(
                text=f"The map of Melbourne train stations has been generated. "
                     f"<a href='{public_url}' target='_blank'>Click here to view the map</a>"
            )

            # Auto-open in browser (local only, optional)
            try:
                webbrowser.open(public_url, new=2)
            except Exception as e:
                logger.warning(f"Could not auto-open browser: {e}")

            return []

        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to generate map", e)
            raise

class ActionGenerateTramMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: TRAM_01
        Name: Tram Stations Map
        Author: AlexT
        Modifier: Juveria Nishath
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_generate_tram_map"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        try:
            # De-duplicate tram stops
            stops_map_df = tram_stops.drop_duplicates(subset=["stop_lat", "stop_lon"])

            # Build map
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)
            marker_cluster = MarkerCluster().add_to(melbourne_map)

            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    popup=f"{row['stop_name']}",
                    tooltip=f"{row['stop_name']}",
                ).add_to(marker_cluster)

            # Save map
            map_filename = "melbourne_tram_stops_map.html"
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # HTTP link served by: python -m http.server 8000
            server_base_url = os.getenv("MAP_SERVER_BASE_URL", "http://localhost:8000")
            public_url = f"{server_base_url}/maps/{map_filename}"

            # Single combined message with one link
            dispatcher.utter_message(
                text=(
                    "The map of Melbourne tram stops has been generated. "
                    f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne tram stops</a>"
                )
            )

            # (Optional) auto-open in browser for local testing
            try:
                webbrowser.open(public_url, new=2)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not auto-open browser: {e}")

            return []

        except Exception as e:
            logging.getLogger(__name__).exception("Failed to generate tram map")
            dispatcher.utter_message(text="An error occurred while generating the tram map.")
            return []
        
class ActionGenerateBusMap(Action):
    ''' -------------------------------------------------------------------------------------------------------
        ID: BUS_01
        Name: Bus Stations Map
        Author: AlexT
        Modifier: Juveria Nishath
        -------------------------------------------------------------------------------------------------------
   '''
    def name(self) -> Text:
        return "action_generate_bus_map"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        try:
            # Ensure bus_stops has the required columns
            stops_map_df = bus_stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]]

            # Build map
            melbourne_map = folium.Map(location=[-37.8136, 144.9631], zoom_start=12)
            for _, row in stops_map_df.iterrows():
                folium.Marker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    popup=f"Stop ID: {row['stop_id']}<br>Stop Name: {row['stop_name']}",
                    tooltip=row["stop_name"],
                ).add_to(melbourne_map)

            # Save map
            map_filename = "melbourne_bus_stops_map.html"
            current_directory = os.getcwd()
            map_folder = os.path.join(current_directory, "maps")
            os.makedirs(map_folder, exist_ok=True)
            map_path = os.path.join(map_folder, map_filename)
            melbourne_map.save(map_path)

            # HTTP link served by: python -m http.server 8000
            server_base_url = os.getenv("MAP_SERVER_BASE_URL", "http://localhost:8000")
            public_url = f"{server_base_url}/maps/{map_filename}"

            # Single combined message with one link
            dispatcher.utter_message(
                text=(
                    "The map of Melbourne bus stops has been generated. "
                    f"<a href='{public_url}' target='_blank'>Click here to view the map of Melbourne bus stops</a>"
                )
            )

            return []

        except Exception as e:
            logging.getLogger(__name__).exception("Failed to generate bus map")
            dispatcher.utter_message(text="An error occurred while generating the bus map.")
            return []

class ActionFindNextTrain(Action):
    ''' -------------------------------------------------------------------------------------------------------
    	ID: REQ_02 implementation
    	Name: Schedule Information
    	Author: AlexT
        Modifier: Andre Nguyen
    	-------------------------------------------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_next_train"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")
            transport_mode = "train"
            logger.info(f"Extracted slots -> station_a: {station_a}, station_b: {station_b}, transport_mode: {transport_mode}")

            query = tracker.latest_message.get('text')
            logger.info(f"User query: {query}")
            
            # Extract stations from query
            extracted_stations = GTFSUtils.extract_stations_from_query(query, stops_df)
            logger.info(f"Extracted stations: {extracted_stations}")

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again with valid station names.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            # Validate stations
            if not station_a:
                dispatcher.utter_message(text="Please specify a starting station.")
                return []

            if not station_b and "to" in query.lower():
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            # Get response from GTFS utils
            response = GTFSUtils.find_next_public_transport_trip(station_a, station_b, "train", stops_df, stop_times_df)
            logger.info(f"Generated response: {response}")

            dispatcher.utter_message(text=response)
            
        except Exception as e:
            logger.error(f"Error in ActionFindNextTrain: {e}")
            import traceback
            traceback.print_exc()
            dispatcher.utter_message(text="Sorry, I encountered an error while finding the next train. Please try again with different station names.")
            return []

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

            stop_a_id = GTFSUtils.get_stop_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_stop_id(station_b, stops_df)

            # stop_a_times = stop_times_df.loc[stop_a_id][['stop_sequence', 'arrival_time']].reset_index()
            # stop_b_times = stop_times_df.loc[stop_b_id][['stop_sequence', 'arrival_time']].reset_index()

            # merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))

            # valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()

            # if valid_trips.empty:
            #     dispatcher.utter_message(text="No direct route found between the two stations.")
            #     return []

            # Andre Nguyen's code
            list_of_child_station_a = GTFSUtils.find_child_station(stop_a_id, stops_df, stop_times_df)
            list_of_child_station_b = GTFSUtils.find_child_station(stop_b_id, stops_df, stop_times_df)
            valid_trips_list = []
            for stop_a_id in list_of_child_station_a:
                stop_a_times = stop_times_df.loc[stop_a_id][['stop_sequence', 'arrival_time']].reset_index()
                # re-create the stop_id column of stop_a
                list_stop_a_id = [stop_a_id] * len(stop_a_times['stop_sequence'])
                stop_a_times['stop_id'] = list_stop_a_id
                for stop_b_id in list_of_child_station_b:
                    stop_b_times = stop_times_df.loc[stop_b_id][['stop_sequence', 'arrival_time']].reset_index()
                    # re-create the stop_id column of stop_b
                    list_stop_b_id = [stop_b_id] * len(stop_b_times['stop_sequence'])
                    stop_b_times['stop_id'] = list_stop_b_id
                    merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))
                    valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()
                    if len(valid_trips) > 0:
                        valid_trips_list.append(valid_trips)

            if len(valid_trips_list) == 0:
                dispatcher.utter_message(text="No direct route found between the two stations.")
                return []
            
            valid_trips = valid_trips_list[0]
            # End of Andre N's code
            valid_trips['arrival_time_a'] = valid_trips['arrival_time_a'].apply(GTFSUtils.parse_time)
            valid_trips['arrival_time_b'] = valid_trips['arrival_time_b'].apply(GTFSUtils.parse_time)
            valid_trips['travel_time'] = (valid_trips['arrival_time_b'] - valid_trips['arrival_time_a']).dt.total_seconds()
            best_trip = valid_trips.loc[valid_trips['travel_time'].idxmin()]

            stop_a_name = stops_df.loc[stops_df['stop_id'] == best_trip['stop_id_a'], 'stop_name'].values[0]
            stop_b_name = stops_df.loc[stops_df['stop_id'] == best_trip['stop_id_b'], 'stop_name'].values[0]
            route_id = trips_df.loc[trips_df['trip_id'] == best_trip['trip_id'], 'route_id'].values[0]
            route_name = routes_df.loc[routes_df['route_id'] == route_id, 'route_long_name'].values[0]
            destination = trips_df.loc[trips_df['trip_id'] == best_trip['trip_id'], 'trip_headsign'].values[0]

            response = f"The best route from {stop_a_name} to {stop_b_name} is on the {route_name} towards {destination}.\n"
            response += f"The trip takes approximately {best_trip['travel_time'] / 60:.2f} minutes."

            # Create the route map given the trip id, including the transfers_df to highlight transfer stations
            hyperlink = GTFSUtils.generate_route_map(
                best_trip['trip_id'], stop_a_name, stop_b_name, stops_df, stop_times_df, dataset_path
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
            station_a = station_a.replace("Railway", "").replace("Station", "").strip()
            station_b = station_b.replace("Railway", "").replace("Station", "").strip()

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

            stop_a_id = GTFSUtils.get_stop_id(station_a, stops_df)
            stop_b_id = GTFSUtils.get_stop_id(station_b, stops_df)

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
        current_addr = tracker.get_slot('address')

        if not location_from:
            current_addr = tracker.get_slot('address')
            station_a = tracker.get_slot('station_a')
            if current_addr:
                location_from = current_addr
            elif station_a:
                location_from = station_a
        
        if not location_to:
            station_b = tracker.get_slot('station_b')
            if station_b:
                location_to = station_b
        
        if not location_from or not location_to:
            dispatcher.utter_message(text="Please provide both starting location and destination in the format: 'How do I get from [location] to [destination]'")
            return []
        best_trip = GTFSUtils.find_pt_route_between_two_address(location_from, location_to, google_api_key)
        if "error" not in best_trip:
            if 'route_description' in best_trip:
                departure_datetime = datetime.now()
                minutes_delta = timedelta(minutes=best_trip['total_time'])
                arrival_datetime = departure_datetime + minutes_delta

                # Format the new datetime to "DD-MM-YY HH-MM-SS"
                formatted_datetime = arrival_datetime.strftime('%d-%m-%y %H:%M:%S')
                dispatcher.utter_message(text=best_trip['route_description'])
                dispatcher.utter_message(text=f"Distance: {round(best_trip['distance_meters'] / 1000, 1)} kilometers")
                dispatcher.utter_message(text=f"Your total travel time: {best_trip['total_time']} min")
                dispatcher.utter_message(text=f"Arrival Time: {formatted_datetime}")
                if 'encoded_polyline' in best_trip:
                    if best_trip['encoded_polyline'] != "":
                        map_file = GTFSUtils.create_polyline_map(best_trip['encoded_polyline'])
                        if os.path.exists(map_file):
                            relative_path = os.path.relpath(map_file, current_dir)
                            map_url = f"http://localhost:8080/{relative_path.replace(os.sep, '/')}"
                            link_message = f'<a href="{map_url}" target="_blank">Click here to view the route map</a>'
                            dispatcher.utter_message(text=link_message, parse_mode="html")
            else:
                dispatcher.utter_message(text=f"I could not find your trip from {location_from} to {location_to}")
        else:
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
                        map_url = f"http://localhost:8080/{relative_path.replace(os.sep, '/')}"

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
Modifier: Juveria Nishath
-------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------
Class: ActionCheckFeature
Purpose: This class handles the intent where a user asks whether a specific feature is available at a particular station.
--------------------------------------------------------------------------------------------------------------------------------------------------
'''
class ActionCheckFeature(Action):
    def name(self) -> Text:
        return "action_check_feature"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # Local imports only (no changes to your global imports)
        import re
        from difflib import get_close_matches
        from rasa_sdk.events import SlotSet

        # ---------- 1) Get raw text + slots ----------
        user_text   = (tracker.latest_message.get("text") or "").strip()
        raw_station = (tracker.get_slot("station_name") or "").strip()
        raw_feature = (tracker.get_slot("feature") or "").strip()

        # If feature slot is empty, try to read it from this message’s entities
        if not raw_feature:
            for ent in (tracker.latest_message.get("entities") or []):
                if (ent.get("entity") or "") == "feature":
                    val = ent.get("value")
                    if val:
                        raw_feature = str(val).strip()
                        break

        # ---------- 2) Normaliser used for BOTH CSV + user text ----------
        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            s = re.sub(r'\brailway\s+station\b', '', s)
            s = re.sub(r'\bstation\b', '', s)
            s = re.sub(r'\brailway\b', '', s)
            s = re.sub(r'[()/_-]', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            return s.strip()

        # Prepare candidate list from CSV (uses prebuilt norm_name if present; else compute)
        if "norm_name" in station_data.columns:
            candidates_series = station_data["norm_name"].astype(str).map(_norm)
        else:
            candidates_series = station_data["Station Name"].astype(str).map(_norm)
        candidates = candidates_series.tolist()

        # ---------- 3) Get station from user text first; then fall back ----------
        if not raw_station:
            # Try GTFS extractor (if available)
            try:
                from .gtfs_utils import GTFSUtils, stops_df
                extracted = GTFSUtils.extract_stations_from_query(user_text, stops_df) or []
                if extracted:
                    raw_station = extracted[0]
            except Exception:
                pass

            # Simple fallback: look for any known station name inside the user text
            if not raw_station and candidates:
                text_norm = _norm(user_text)
                hits = [c for c in candidates if c and c in text_norm]
                if hits:
                    raw_station = hits[0]

        # ---------- 4) Validate missing pieces ----------
        if not raw_station and not raw_feature:
            dispatcher.utter_message(text="Please specify both the station name and the feature you are asking about.")
            return []
        if not raw_station:
            dispatcher.utter_message(text="Please tell me the station name.")
            return []
        if not raw_feature:
            dispatcher.utter_message(text=f"What feature do you want to check at {raw_station}? (e.g., lifts, restroom)")
            return []

        station_key = _norm(raw_station)
        feature_key = (raw_feature or "").lower().strip()

        if not candidates:
            dispatcher.utter_message(text="Station list is unavailable right now. Please try again later.")
            return []

        # ---------- 5) Exact or fuzzy match the station (typo tolerant) ----------
        if station_key not in candidates:
            best = get_close_matches(station_key, candidates, n=1, cutoff=0.80)
            if best:
                station_key = best[0]
            else:
                dispatcher.utter_message(
                    text=f"Sorry, I couldn’t find “{raw_station}”. Please check the spelling or try another station."
                )
                return []

        idxs = candidates_series[candidates_series == station_key].index
        if len(idxs) == 0:
            dispatcher.utter_message(text=f"Sorry, I couldn’t find “{raw_station}”.")
            return []

        row = station_data.loc[idxs[0]]
        display_name = str(row["Station Name"]).title()

        # ---------- 6) Feature → CSV column mapping (includes washrooms) ----------
        feature_mapping = {
            "escalators": "Escalators", "escalator": "Escalators",
            "lift": "Lift", "lifts": "Lift", "elevator": "Lift", "elevators": "Lift",
            "ramps": "Station access", "ramp": "Station access", "access": "Station access",
            "parking": "Parking", "car park": "Parking",
            "restroom": "Toilet", "restrooms": "Toilet",
            "toilet": "Toilet", "toilets": "Toilet",
            "bathroom": "Toilet", "bathrooms": "Toilet",
            "washroom": "Toilet", "washrooms": "Toilet",
            "tactile edges": "Tactile edges", "tactile": "Tactile edges",
            "hearing loop": "Hearing Loop", "hearing loops": "Hearing Loop",
            "info screens": "Info screens", "information screens": "Info screens",
            "shelter": "Shelter", "low platform": "Low platform",
            "path widths": "Path Widths",
            "pick up / drop off": "Pick up / Drop off", "pick-up/drop-off": "Pick up / Drop off",
        }

        column_name = feature_mapping.get(feature_key)
        if not column_name:
            supported = ", ".join(sorted(set(feature_mapping.keys())))
            dispatcher.utter_message(
                text=f"Sorry, I don't have information about '{raw_feature}'. Try one of: {supported}."
            )
            return []
        if column_name not in station_data.columns:
            dispatcher.utter_message(text=f"Sorry, I don't track '{raw_feature}' for stations yet.")
            return []

        # ---------- 7) Answer ----------
        val = str(row[column_name]).strip().lower()
        has_it = not (val in ("", "nan", "no", "false", "0"))
        dispatcher.utter_message(
            text=f"{'Yes' if has_it else 'No'}, {display_name} {'has' if has_it else 'does not have'} {raw_feature}."
        )

        # Clear slots for next turn
        return [SlotSet("station_name", None), SlotSet("feature", None)]
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
	Name: Traffic details for specific location
	Author: Awanish
	-------------------------------------------------------------------------------------------------------
'''
class ActionFetchTraffic(Action):
    def name(self) -> Text:
        return "action_fetch_traffic"

    def run(self, dispatcher, tracker, domain):
        source = tracker.get_slot("station_a")
        
        source_coords = geocode_address(source)
        #source_coords = getAddressLatLong( address = source)
        #destination_coords = getAddressLatLong( address = destination )

        api_key = "cBr8gVX64Y3q18xGfTKnxHvsGl7AcnMw"
        
        traffic_details_location = get_traffic_details(api_key, source_coords)
        traffic_status_start = get_traffic_status(
                traffic_details_location['current_speed'], traffic_details_location['free_flow_speed']
            )
        dispatcher.utter_message(
            text=f"Traffic is {traffic_status_start} in {source} with a current speed of {traffic_details_location['current_speed']} km/h."
        )

''' -------------------------------------------------------------------------------------------------------
	
	Name: Traffic details for two location
	Author: Awanish
	-------------------------------------------------------------------------------------------------------
'''
class ActionFetchTrafficLocation(Action):
    def name(self) -> Text:
        return "action_fetch_traffic_location"

    def run(self, dispatcher, tracker, domain):
        source = tracker.get_slot("station_a")
        destination = tracker.get_slot("station_b")
        print(f"from: {source}")
        print(f"to: {destination}")
        
        source_coords = geocode_address(source)
        destination_coords = geocode_address(destination)
        #source_coords = getAddressLatLong( address = source)
        #destination_coords = getAddressLatLong( address = destination )
        
        api_key = "cBr8gVX64Y3q18xGfTKnxHvsGl7AcnMw"
        route_data = fetch_route(source_coords, destination_coords, api_key)
       
        if "error" in route_data:
            dispatcher.utter_message(text=f"Error: {route_data['error']}")
        else:
            routes = route_data.get("routes", [])
            if not routes:
                dispatcher.utter_message(text="No routes found.")
            else:
                route = routes[0]
                summary = route.get("summary", {})
                distance = summary.get("lengthInMeters", 0) / 1000
                travel_time = summary.get("travelTimeInSeconds", 0) / 60

                #start_location = tuple(map(float, source_coords.split(',')))  # Melbourne CBD
                traffic_details_start = get_traffic_details(api_key, source_coords)
                
                #traffic details for the destination location
                #destination_location = tuple(map(float, destination_coords.split(',')))  # Albert Park
                traffic_details_destination = get_traffic_details(api_key, destination_coords)
               
                if traffic_details_start and traffic_details_destination:
                    traffic_status_start = get_traffic_status(
                        traffic_details_start['current_speed'], traffic_details_start['free_flow_speed']
                    )
                    traffic_status_destination = get_traffic_status(
                        traffic_details_destination['current_speed'], traffic_details_destination['free_flow_speed']
                    )
                dispatcher.utter_message(
                    text=f"Traffic is {traffic_status_start} from {source} to {destination} with a current speed of {traffic_details_start['current_speed']} km/h. Estimated travel time is {travel_time:.2f} minutes for a distance of {distance:.2f} km."
                )
def fetch_route(source_coords, destination_coords, api_key):
    base_url = "https://api.tomtom.com/routing/1/calculateRoute"
    source = f"{source_coords[0]},{source_coords[1]}"
    destination = f"{destination_coords[0]},{destination_coords[1]}"
    params = {
        "routeType": "fastest",
        "traffic": "true",
        "travelMode": "bus", # or car
        "key": api_key
    }
    url = f"{base_url}/{source}:{destination}/json"
    
    try:
        response = requests.get(url, params = params)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API call failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_traffic_details(api_key, location):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    latitude, longitude = location
    params = {
        "point": f"{latitude},{longitude}",
        "unit": "KMPH",
        "openLr": "false",
        "key": api_key
    }
    try:
        response = requests.get(url, params = params)
        response.raise_for_status()
        data = response.json()
        traffic_info = {
            "current_speed": data['flowSegmentData']['currentSpeed'],
            "free_flow_speed": data['flowSegmentData']['freeFlowSpeed'],
            "confidence": data['flowSegmentData']['confidence'] * 100,
        }
        return traffic_info
    except Exception as e:
        return None

def get_traffic_status(current_speed, free_flow_speed):
    speed_ratio = current_speed / free_flow_speed
    if speed_ratio >= 0.8:
        return "light"
    elif 0.6 <= speed_ratio < 0.8:
        return "moderate"
    else:
        return "heavy"

#Function to geocode an address using Nominatim
def geocode_address(address):
    geolocator = Nominatim(user_agent="andre_nguyen")
    
    #Define the bounding box for Melbourne
    melbourne_bbox = [(-38.433859,144.593741), (-37.511274,145.512529)]
    
    #Geocode the address within the Melbourne bounding box
    location = geolocator.geocode(address, viewbox=melbourne_bbox, bounded=True)
    
    if location:
        return location.latitude, location.longitude
    else:
        print("Address not found within Melbourne.")
        return None

'''
-------------------------------------------------------------------------------------------------------
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
        
        if not location_from:
            current_addr = tracker.get_slot('address')
            station_a = tracker.get_slot('station_a')
            if current_addr:
                location_from = current_addr
            elif station_a:
                location_from = station_a
        
        if not location_to:
            station_b = tracker.get_slot('station_b')
            if station_b:
                location_to = station_b

        logger.info(f"Final location_from: {location_from}")
        logger.info(f"Final location_to: {location_to}")
        logger.info(f"Slot extraction took {time.time() - slots_start:.2f} seconds")
        
        if not location_from or not location_to:
            error_msg = f"Missing {'origin' if not location_from else 'destination'} location"
            logger.error(f"Validation failed: {error_msg}")
            dispatcher.utter_message(text=f"Please provide both starting location and destination. {error_msg}.")
            return []
        
        # provide region
        location_from += ", Australia"
        location_to += ", Australia"
        best_trip = GTFSUtils.find_pt_route_between_two_address(location_from, location_to, google_api_key)
        if "error" not in best_trip:
            # Include nation in the location name
            if 'route_description' in best_trip:
                departure_datetime = datetime.now()
                minutes_delta = timedelta(minutes=best_trip['total_time'])
                arrival_datetime = departure_datetime + minutes_delta

                # Format the new datetime to "DD-MM-YY HH-MM-SS"
                formatted_datetime = arrival_datetime.strftime('%d-%m-%y %H:%M:%S')
                dispatcher.utter_message(text=best_trip['route_description'])
                dispatcher.utter_message(text=f"Distance: {round(best_trip['distance_meters'] / 1000, 1)} kilometers")
                dispatcher.utter_message(text=f"Your total travel time: {best_trip['total_time']} min")
                dispatcher.utter_message(text=f"Arrival Time: {formatted_datetime}")
                if 'encoded_polyline' in best_trip:
                    if best_trip['encoded_polyline'] != "":
                        map_file = GTFSUtils.create_polyline_map(best_trip['encoded_polyline'])
                        if os.path.exists(map_file):
                            relative_path = os.path.relpath(map_file, current_dir)
                            map_url = f"http://localhost:8080/{relative_path.replace(os.sep, '/')}"
                            link_message = f'<a href="{map_url}" target="_blank">Click here to view the route map</a>'
                            dispatcher.utter_message(text=link_message, parse_mode="html")

            else:
                dispatcher.utter_message(text=f"I could not find your trip from {location_from} to {location_to}")
        else:
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
                        map_url = f"http://localhost:8080/{relative_path.replace(os.sep, '/')}"
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
    tram routing with transfers
    by: JubalK
    -----------------------------------------------------------------------
    '''
    def name(self) -> Text:
        return "action_find_tram_route_with_transfers"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract transport mode (default to train if not provided) # Extract user input and slots
            station_a = tracker.get_slot("station_a")
            station_b = tracker.get_slot("station_b")

            logger.info(
                f"Extracted slots -> station_a: {station_a}, station_b: {station_b}")

            query = tracker.latest_message.get('text')
            print(f"User query: {query}")

            extracted_stations = GTFSUtils.extract_stations_from_query(query, tram_stops)

            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a, station_b = extracted_stations[0], extracted_stations[1]

            print(f"Station A: {station_a}")
            print(f"Station B: {station_b}")

            tram_stop_times.sort_index(inplace=True)

            stop_a_id = GTFSUtils.get_stop_id(station_a, tram_stops) #get stop id for starting
            stop_b_id = GTFSUtils.get_stop_id(station_b, tram_stops) #get stop id for destination

            print(f"stop id for stop a: {stop_a_id}")
            print(f"stop id for stop b: {stop_b_id}")

            trip_id = tram_stop_times.loc[stop_b_id].index.get_level_values('trip_id').unique()[0]

            queue = deque([(stop_a_id, [stop_a_id])])
            visited = set()

            if stop_a_id is None or stop_b_id is None:
                dispatcher.utter_message(text=f"Sorry, I couldn't find stop IDs for {station_a} and {station_b}.")
                return []

            best_route = None
            destination_found = False
            tram_stop_times_reset = tram_stop_times.reset_index()

            stop_route_data = tram_stop_times_reset.merge(
                tram_trips[['trip_id', 'route_id']],  on='trip_id'
            )[['stop_id', 'route_id']] # Ensure tram_trips has 'trip_id' and 'route_id'

            print(f"stop_route data with the station_b id: {stop_route_data[stop_route_data['stop_id'] == stop_b_id]}")

            route_stops = {
                route_id: set(group['stop_id'])
                for route_id, group in stop_route_data.groupby('route_id')
            }

            while queue:
                current_stop_id, path = queue.popleft()  # Dequeue the current station and path

                if current_stop_id == stop_b_id:
                    # Destination reached, store the best route
                    best_route = path
                    print(f"Destination reached: {station_b}")
                    destination_found = True
                    break

                if current_stop_id in visited:
                    print(f"Station {current_stop_id} already processed, skipping.")
                    continue

                visited.add(current_stop_id)
                print(f"Processing station: {current_stop_id}")

                #current_stop_id = GTFSUtils.get_stop_id(current_stop_id, tram_stops)
                current_routes = {route_id for route_id, stops in route_stops.items() if current_stop_id in stops}
                if current_stop_id is None:
                    continue

                for route_id in current_routes:
                    if destination_found:
                        break

                    for next_stop_id  in route_stops[route_id]:
                        if destination_found:
                            break

                        if next_stop_id == stop_b_id:
                            # Direct route found
                            best_route = path + [next_stop_id ]
                            print(f"Direct route found to destination: {station_b}")
                            queue.clear()
                            destination_found = True
                            break

                        if next_stop_id not in visited:
                            # Add the next station to the queue with the updated path
                            queue.append((next_stop_id, path + [next_stop_id]))
                            print(f"Queued station: {next_stop_id}")

            if not best_route:
                dispatcher.utter_message(
                    text=f"Sorry, I couldn't find a suitable route from {station_a} to {station_b}.")
                return []

            best_route_names = [tram_stops.loc[tram_stops['stop_id'] == stop_id, 'stop_name'].values[0] for stop_id in best_route]
            response = f"The best route from {station_a} to {station_b} involves the following transfers: {', '.join(best_route_names)}.\n"

            dispatcher.utter_message(text=response)


        except Exception as e:
            GTFSUtils.handle_error(dispatcher, logger, "Failed to find the best route with transfers", e)
            raise

        return []


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

            stop_a_id = GTFSUtils. get_stop_id(station_a, tram_stops)
            stop_b_id = GTFSUtils. get_stop_id(station_b, tram_stops) if station_b else None

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
# class ActionFindNearestStation(Action):
#     ''' -------------------------------------------------------------------------------------------------------
#         ID: TRAIN
#         Name: Find Nearest Station
#         Author: RossP
#         -------------------------------------------------------------------------------------------------------
#    '''
#     def name(self) -> Text:
#         return "action_find_nearest_station"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         entities = tracker.latest_message.get("entities", [])

#         unique_val = []
#         seen_val = set()
        
#         for entity in entities:
#             if entity['value'] not in seen_val:
#                 unique_val.append(entity['value'])
#                 seen_val.add(entity['value']) 
#             if len(unique_val) == 2:
#                 break   
         
#         address_entity = ", ".join(unique_val) 
#         logger.debug(address_entity)
        
#         #get lat and long of location form google API
#         addll = GTFSUtils.getAddressLatLong(address = address_entity)
#         logger.debug(addll)

#         #check distance to all stations
#         closStat = GTFSUtils.checkDistancetoAllStation(addll['latitude'],addll['longitude'])
                        
        
#         if address_entity:
#                 dispatcher.utter_message(text = f"The closest station to {address_entity} is {closStat['closest_station_name']}")
#         else: 
#             dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
#         return []

# class ActionFindNearestTramStop(Action):
#     ''' -------------------------------------------------------------------------------------------------------
#     ID: TRAM
#     Name: Find Nearest Tram Stop
#     Author: RossP
#     -------------------------------------------------------------------------------------------------------
#     ''' 
#     def name(self) -> Text:
#         return "action_find_nearest_tram_stop"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         entities = tracker.latest_message.get("entities", [])

#         unique_val = []
#         seen_val = set()
        
#         for entity in entities:
#             if entity['value'] not in seen_val:
#                 unique_val.append(entity['value'])
#                 seen_val.add(entity['value']) 
#             if len(unique_val) == 2:
#                 break   
         
#         address_entity = ", ".join(unique_val) 
#         logger.debug(address_entity)
        
#         #get lat and long of location form google API
#         addll = GTFSUtils.getAddressLatLong(address = address_entity)
#         logger.debug(addll)

#         #check distance to all stations
#         closStat = GTFSUtils.checkDistancetoAllTramsStops(addll['latitude'],addll['longitude'])
                        
        
#         if address_entity:
#                 dispatcher.utter_message(text = f"The closest Tram Stop to {address_entity} is {closStat['closest_station_name']}")
#         else: 
#             dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
#         return []

# class ActionFindNearestBusStop(Action):    
#     ''' -------------------------------------------------------------------------------------------------------
#     ID: BUS
#     Name: Find Nearest Bus Stop
#     Author: RossP
#     -------------------------------------------------------------------------------------------------------
#     '''
#     def name(self) -> Text:
#         return "action_find_nearest_bus_stop"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         entities = tracker.latest_message.get("entities", [])

#         unique_val = []
#         seen_val = set()
        
#         for entity in entities:
#             if entity['value'] not in seen_val:
#                 unique_val.append(entity['value'])
#                 seen_val.add(entity['value']) 
#             if len(unique_val) == 2:
#                 break   
         
#         address_entity = ", ".join(unique_val) 
#         logger.debug(address_entity)
        
#         #get lat and long of location form google API
#         addll = GTFSUtils.getAddressLatLong(address = address_entity)
#         logger.debug(addll)

#         #check distance to all stations
#         closStat = GTFSUtils.checkDistancetoAllBusStops(addll['latitude'],addll['longitude'])
                        
        
#         if address_entity:
#                 dispatcher.utter_message(text = f"The closest Bus Stop to {address_entity} is {closStat['closest_station_name']}")
#         else: 
#             dispatcher.utter_message(text = 'Sorry Address not found please try again')
        
        
#         return []
    
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
# Juveria Nishath Actions start
class ActionDrivingTimeByCar(Action):
    ''' -------------------------------------------------------------------------------------------------------
        Name: Driving time by car (TomTom)
        Author: Juveria Nishath
        -------------------------------------------------------------------------------------------------------
        What it does:
            • Parses origin/destination from slots or “from … to …” text
            • Geocodes both ends via TomTom Search
            • Calls TomTom Routing (fastest, traffic on) to get ETA + distance

    ------------------------------------------------------------------------------------------------------- '''

    def name(self) -> Text:
        return "action_driving_time_by_car"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        try:
            # (1) Grab the raw user message text (handles full addresses, commas, line breaks)
            user_text = (tracker.latest_message or {}).get("text", "") or ""

            # (2) Extract origin/destination with strict preference: regex > entities > old slots
            # (2a) Regex: capture "from ... to ..." even across newlines
            origin_rgx = dest_rgx = None
            m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)", user_text, flags=re.I | re.S)
            if m:
                origin_rgx = m.group(1).strip()
                dest_rgx   = m.group(2).strip()

            # (2b) Latest entities (if your NLU tagged them)
            ents = (tracker.latest_message or {}).get("entities", []) or []
            origin_ent = next(
                (e.get("value") for e in ents if e.get("entity") in ("source", "location_from", "station_a")),
                None
            )
            dest_ent = next(
                (e.get("value") for e in ents if e.get("entity") in ("destination", "location_to", "station_b")),
                None
            )

            # (2c) Build final origin/dest, preferring regex > entities > prior slots
            origin = (origin_rgx or origin_ent
                      or tracker.get_slot("source")
                      or tracker.get_slot("location_from")
                      or tracker.get_slot("station_a") or "").strip()

            dest = (dest_rgx or dest_ent
                    or tracker.get_slot("destination")
                    or tracker.get_slot("location_to")
                    or tracker.get_slot("station_b") or "").strip()

            # (3) If still missing any side, ask cleanly and stop (no exceptions)
            if not origin or not dest:
                dispatcher.utter_message(
                    text='Please provide both the starting point and destination, e.g., '
                         '"by car from 4 Anderson St Yarraville to 1016 Morris Rd Truganina".'
                )
                return []

            # (4) Pull TomTom key from environment; fail fast if not set (no 503 crashes)
            api_key = os.getenv("TOMTOM_API_KEY")
            if not api_key:
                dispatcher.utter_message(text="TomTom API key is not configured on the server.")
                return []

            # (5) Geocode BOTH ends with Melbourne bias + VIC preference (to avoid NSW/QLD/TAS)
            MEL_CBD_LAT, MEL_CBD_LON = -37.8136, 144.9631   # Melbourne CBD
            MEL_RADIUS_KM = 150

            # First attempt with CBD bias (keeps ambiguous names in metro VIC)
            o = tt_geocode(origin, api_key,
                           country_set="AU", bias_lat=MEL_CBD_LAT, bias_lon=MEL_CBD_LON, radius_km=MEL_RADIUS_KM)
            d = tt_geocode(dest,   api_key,
                           country_set="AU", bias_lat=MEL_CBD_LAT, bias_lon=MEL_CBD_LON, radius_km=MEL_RADIUS_KM)

            # (5b) If either side failed, retry with explicit ", VIC" suffix (no extra kwargs)
            if not o:
                o = tt_geocode(f"{origin}, VIC", api_key,
                               country_set="AU", bias_lat=MEL_CBD_LAT, bias_lon=MEL_CBD_LON, radius_km=MEL_RADIUS_KM)
            if not d:
                d = tt_geocode(f"{dest}, VIC", api_key,
                               country_set="AU", bias_lat=MEL_CBD_LAT, bias_lon=MEL_CBD_LON, radius_km=MEL_RADIUS_KM)

            # (5c) Still missing? Tell the user which side failed
            if not o or not d:
                which = "origin" if not o else "destination"
                dispatcher.utter_message(text=f"Sorry, I couldn't locate the {which} on the map.")
                return []

            o_lat, o_lon, o_label = o
            d_lat, d_lon, d_label = d

            # (6) Request the fastest *car* route with traffic ON (current conditions)
            summary = tt_route(o_lat, o_lon, d_lat, d_lon, api_key)
            if not summary:
                dispatcher.utter_message(text="I couldn't fetch the driving time right now.")
                return []

            # (7) Format a friendly message (ETA with/without traffic + distance)
            travel = summary.get("travelTimeInSeconds")                  # with live traffic
            free   = summary.get("noTrafficTravelTimeInSeconds")         # free-flow (no traffic)
            delay  = summary.get("trafficDelayInSeconds")                # pure delay component
            hist   = summary.get("historicTrafficTravelTimeInSeconds")   # typical traffic (optional)
            live   = summary.get("liveTrafficIncidentsTravelTimeInSeconds")  # incidents component (optional)
            dist_m = summary.get("lengthInMeters")
            arrive = summary.get("arrivalTime")  # ISO 8601 string; optional

            parts = [f"Driving time from {o_label} to {d_label} (now): {fmt_time(travel)}"]

            if free is not None:
                parts.append(f"free-flow: {fmt_time(free)}")
            if delay is not None:
                parts.append(f"traffic delay: {fmt_time(delay)}")
            if hist is not None:
                parts.append(f"typical traffic: {fmt_time(hist)}")
            if live is not None:
                parts.append(f"incidents time: {fmt_time(live)}")
            if dist_m is not None:
                parts.append(f"distance: {fmt_km(dist_m)}")
            if arrive:
                parts.append(f"ETA arrival: {arrive}")

            msg = "; ".join(parts) + "."
            dispatcher.utter_message(text=msg)
            return []

        except Exception as e:
            logger.exception("TomTom driving-time action failed: %s", e)
            dispatcher.utter_message(text="Sorry, I couldn't compute the driving time due to an error.")
            return []