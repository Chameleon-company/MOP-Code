import spacy
import folium
import os
import pandas as pd
import logging
from typing import Any, Text, Dict, List
from fuzzywuzzy import process, fuzz
from datetime import datetime
from rasa_sdk.events import SlotSet
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

''' -------------------------------------------------------------------------------------------------------	
	Author: AlexT
	- Load Load GTFS data into DataFrames
	- Normalise the stop names
	- Indexing to speed up searching performance stop_times
	-------------------------------------------------------------------------------------------------------
'''
# Load GTFS data into DataFrames
#dataset_path = r'C:\Users\logan\MOP Clone\MOP-Code\artificial-intelligence\chatbot\code\mpt_bot\gtfs\2'

# Load GTFS data into DataFrames using a relative path
dataset_path = os.path.join('gtfs', '2')

stops_df = pd.read_csv(os.path.join(dataset_path, 'stops.txt'))
routes_df = pd.read_csv(os.path.join(dataset_path, 'routes.txt'))
trips_df = pd.read_csv(os.path.join(dataset_path, 'trips.txt'))
stop_times_df = pd.read_csv(os.path.join(dataset_path, 'stop_times.txt'))
calendar_df = pd.read_csv(os.path.join(dataset_path, 'calendar.txt'))

# Normalize the stop names
stops_df['normalized_stop_name'] = stops_df['stop_name'].str.lower()

# Ensure the stop_times DataFrame has the necessary columns and is indexed correctly
expected_columns = ['stop_id', 'trip_id', 'arrival_time', 'departure_time']

if all(col in stop_times_df.columns for col in expected_columns):
    stop_times_df.set_index(['stop_id', 'trip_id'], inplace=True)
else:
    print("Error: Expected columns are not present in the DataFrame.")
    print("Available columns:", stop_times_df.columns)

'''-------------------------------------------------------------------------------------------------------'''
''' -------------------------------------------------------------------------------------------------------	
	Generic methods can be use in all actions
	Author: AlexT
		- find_station_name
		- convert_gtfs_time
	-------------------------------------------------------------------------------------------------------
	# Test the function with different inputs
    # user_input = "Footscray"
    # best_station = find_station_name(user_input, stops_df)
    # print("Best matched station:", best_station)
'''
class GTFSUtils:
    """Utility class for GTFS-related functions."""
    @staticmethod
    def find_station_name(user_input: str, stops_df: pd.DataFrame) -> str:
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
    def convert_gtfs_time(gtfs_time: str) -> str:
        """Convert GTFS time format to a more readable format."""
        hours, minutes, seconds = map(int, gtfs_time.split(':'))
        if hours >= 24:
            hours -= 24
            return f"{hours:02}:{minutes:02}:{seconds:02} (next day)"
        return f"{hours:02}:{minutes:02}:{seconds:02}"

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
            doc = nlp(query)
            potential_stations = [ent.text for ent in doc.ents]

            if not potential_stations:
                potential_stations = [GTFSUtils.find_station_name(query, stops_df)]

            extracted_stations = []
            for station in potential_stations:
                matched_station = GTFSUtils.find_station_name(station, stops_df)
                if matched_station:
                    extracted_stations.append(matched_station)

            if len(extracted_stations) == 0:
                dispatcher.utter_message(text="Sorry, I couldn't find any stations in your query. Please try again.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1] if len(extracted_stations) > 1 else None

            if not station_a or (not station_b and "to" in query.lower()):
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            stop_a_id = stops_df[stops_df['stop_name'] == station_a]['stop_id'].values[0]

            if station_b:
                stop_b_id = stops_df[stops_df['stop_name'] == station_b]['stop_id'].values[0]

            current_time = datetime.now().strftime('%H:%M:%S')

            if not isinstance(stop_times_df.index, pd.MultiIndex):
                stop_times_df.set_index(['stop_id', 'trip_id'], inplace=True, drop=False)

            if not station_b:
                trips_from_station = stop_times_df.loc[stop_a_id]
                trips_from_station = trips_from_station[trips_from_station['departure_time'] >= current_time]
                trips_from_station = trips_from_station.sort_values('departure_time').drop_duplicates(subset=['departure_time'])

                if not trips_from_station.empty:
                    next_trips = trips_from_station[['departure_time']].head(5)
                    response = f"Upcoming train schedules from {station_a}:\n"
                    for idx, row in next_trips.iterrows():
                        departure_time = GTFSUtils.convert_gtfs_time(row['departure_time'])
                        response += f"- Train at {departure_time}\n"
                else:
                    response = f"No upcoming trains found from {station_a}."
            else:
                trips_from_station_a = stop_times_df.loc[stop_a_id].reset_index()
                trips_to_station_b = stop_times_df.loc[stop_b_id].reset_index()

                future_trips = trips_from_station_a[trips_from_station_a['departure_time'] >= current_time]['trip_id'].unique()
                valid_trips = trips_to_station_b[trips_to_station_b['trip_id'].isin(future_trips)]

                if not valid_trips.empty:
                    next_trip = valid_trips.iloc[0]
                    next_trip_time = trips_from_station_a[
                        (trips_from_station_a['trip_id'] == next_trip['trip_id'])
                    ]['departure_time'].values[0]
                    next_trip_time = GTFSUtils.convert_gtfs_time(next_trip_time)
                    response = f"The next train from {station_a} to {station_b} leaves at {next_trip_time}."
                else:
                    response = f"No upcoming trains found from {station_a} to {station_b}."

            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(text=f"Failed to find the next train: {str(e)}")
            raise

        return []
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
            dispatcher.utter_message(text=f"Failed to generate map: {str(e)}")
            raise

        return []
''' -------------------------------------------------------------------------------------------------------
	ID: REQ_01 implementation
	Name: Basic Route Planning
	Author: AlexT
	-------------------------------------------------------------------------------------------------------
'''
class ActionFindBestRoute(Action):

    def name(self) -> Text:
        return "action_find_best_route"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Extract user query
            query = tracker.latest_message.get('text')
            doc = nlp(query)
            potential_stations = [ent.text for ent in doc.ents]

            # Use custom matching if spaCy doesn't find entities
            if not potential_stations:
                potential_stations = [GTFSUtils.find_station_name(query, stops_df)]

            extracted_stations = []
            for station in potential_stations:
                matched_station = GTFSUtils.find_station_name(station, stops_df)
                if matched_station:
                    extracted_stations.append(matched_station)

            # Check if at least two stations were found
            if len(extracted_stations) < 2:
                dispatcher.utter_message(text="Please specify both the starting and destination stations.")
                return []

            station_a = extracted_stations[0]
            station_b = extracted_stations[1]

            # Find the best route
            stop_a_id = stops_df[stops_df['stop_name'] == station_a]['stop_id'].values[0]
            stop_b_id = stops_df[stops_df['stop_name'] == station_b]['stop_id'].values[0]

            stop_a_times = stop_times_df[stop_times_df['stop_id'] == stop_a_id][['stop_sequence', 'arrival_time']].reset_index()
            stop_b_times = stop_times_df[stop_times_df['stop_id'] == stop_b_id][['stop_sequence', 'arrival_time']].reset_index()

            # Merge data to find common trips
            merged = pd.merge(stop_a_times, stop_b_times, on='trip_id', suffixes=('_a', '_b'))
            valid_trips = merged[merged['stop_sequence_a'] < merged['stop_sequence_b']].copy()

            if valid_trips.empty:
                dispatcher.utter_message(text="No direct route found between the two stations.")
                return []

            # Calculate travel time and find the best trip
            valid_trips['arrival_time_a'] = valid_trips['arrival_time_a'].apply(parse_time)
            valid_trips['arrival_time_b'] = valid_trips['arrival_time_b'].apply(parse_time)
            valid_trips['travel_time'] = (valid_trips['arrival_time_b'] - valid_trips['arrival_time_a']).dt.total_seconds()

            best_trip = valid_trips.loc[valid_trips['travel_time'].idxmin()]
            route_id = trips_df[trips_df['trip_id'] == best_trip['trip_id']]['route_id'].values[0]
            route_name = routes_df[routes_df['route_id'] == route_id]['route_long_name'].values[0]

            response = f"The best route from {station_a} to {station_b} is on route {route_name}, trip ID {best_trip['trip_id']}, taking approximately {best_trip['travel_time'] / 60:.2f} minutes."
            dispatcher.utter_message(text=response)

        except Exception as e:
            dispatcher.utter_message(text=f"Failed to find the best route: {str(e)}")
            raise

        return []
    


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

        # Get the two most recent user messages
        user_messages = [event['text'] for event in tracker.events if event.get('event') == 'user']
        user_location = user_messages[-2] if len(user_messages) >= 2 else None
        destination = user_messages[-1] if user_messages else None

        if not user_location or not destination:
            dispatcher.utter_message(text="I couldn't understand the location or destination. Please provide both.")
            return []

        script_path = r"C:\Users\logan\Desktop\Uni\Team proj\basemodelintegratedwithmap\actions\userlocationmaps_executablepassingactions.py"
        map_file_path = "map.html"  

        try:
            #Use subprocess to run the script and capture output
            result = subprocess.run([sys.executable, script_path, user_location, destination], 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)

            if result.stdout.strip():
                if "Address not found" in result.stdout or "Current location could not be determined" in result.stdout:
                    dispatcher.utter_message(text="It seems the location or destination could not be found. Please check your input and try again.")
                else:
                    dispatcher.utter_message(text=f"The direction script has been executed successfully. Here's the output:\n{result.stdout}")

                    #Check if the map file was generated
                    if os.path.exists(map_file_path):
                        dispatcher.utter_message(text="A map has been generated and should open in your default web browser.")
                    else:
                        dispatcher.utter_message(text="The script ran successfully, but no map was generated.")

            else:
                dispatcher.utter_message(text="The direction script has been executed successfully, but no output was produced.")

        except subprocess.CalledProcessError as e:
            dispatcher.utter_message(text=f"An error occurred while running the script: {e.stderr}")
            logger.error(f"Script execution failed: {e.stderr}")
        except Exception as e:
            dispatcher.utter_message(text=f"An unexpected error occurred: {str(e)}")
            logger.error(f"Exception occurred: {str(e)}")

        return [SlotSet("user_location", user_location), SlotSet("destination", destination)]
