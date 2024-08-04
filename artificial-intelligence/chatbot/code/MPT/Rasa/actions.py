from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import pandas as pd

# Load GTFS data into DataFrames
dataset_path = 'C:/Users/Alex.Truong/PycharmProjects/pythonProject/MPT/ds/gtfs/2'
stops_df = pd.read_csv(f'{dataset_path}/stops.txt')
routes_df = pd.read_csv(f'{dataset_path}/routes.txt')
trips_df = pd.read_csv(f'{dataset_path}/trips.txt')
stop_times_df = pd.read_csv(f'{dataset_path}/stop_times.txt')
calendar_df = pd.read_csv(f'{dataset_path}/calendar.txt')


class ActionFindNearbyStation(Action):

    def name(self) -> str:
        return "action_find_nearby_station"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Mock logic for finding a nearby station
        nearby_station = stops_df.iloc[0]['stop_name']
        dispatcher.utter_message(text=f"The nearest station is {nearby_station}.")
        return []


class ActionSearchByName(Action):

    def name(self) -> str:
        return "action_search_by_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        station_name = tracker.get_slot('station_name')
        if station_name:
            station_info = stops_df[stops_df['stop_name'].str.contains(station_name, case=False, na=False)]
            if not station_info.empty:
                dispatcher.utter_message(
                    text=f"{station_name} is located at latitude {station_info.iloc[0]['stop_lat']} and longitude {station_info.iloc[0]['stop_lon']}.")
            else:
                dispatcher.utter_message(text=f"Sorry, I couldn't find information for {station_name}.")
        return []


class ActionSearchByStopID(Action):

    def name(self) -> str:
        return "action_search_by_stopid"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        stop_id = tracker.get_slot('stop_id')
        if stop_id:
            stop_info = stops_df[stops_df['stop_id'] == int(stop_id)]
            if not stop_info.empty:
                dispatcher.utter_message(
                    text=f"StopID {stop_id} corresponds to {stop_info.iloc[0]['stop_name']}, located at latitude {stop_info.iloc[0]['stop_lat']} and longitude {stop_info.iloc[0]['stop_lon']}.")
            else:
                dispatcher.utter_message(text=f"Sorry, I couldn't find information for StopID {stop_id}.")
        return []


class ActionStationDetails(Action):

    def name(self) -> str:
        return "action_station_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        station_name = tracker.get_slot('station_name')
        if station_name:
            station_info = stops_df[stops_df['stop_name'].str.contains(station_name, case=False, na=False)]
            if not station_info.empty:
                # Mock details for facilities
                facilities = "parking, restrooms, elevators"
                dispatcher.utter_message(text=f"{station_name} has the following facilities: {facilities}.")
            else:
                dispatcher.utter_message(text=f"Sorry, I couldn't find details for {station_name}.")
        return []


class ActionDirections(Action):

    def name(self) -> str:
        return "action_directions"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        station_name = tracker.get_slot('station_name')
        if station_name:
            # Mock logic for providing directions
            directions = f"Head north and turn right to reach {station_name}."
            dispatcher.utter_message(text=directions)
        return []


class ActionFilterStations(Action):

    def name(self) -> str:
        return "action_filter_stations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        zone = tracker.get_slot('zone')
        route_name = tracker.get_slot('route_name')

        if zone:
            stations_in_zone = stops_df[stops_df['stop_name'].str.contains(f'Zone {zone}', case=False, na=False)]
            station_list = ", ".join(stations_in_zone['stop_name'].tolist())
            dispatcher.utter_message(text=f"Stations in Zone {zone}: {station_list}")

        elif route_name:
            # Assuming we have a route-station mapping (simplified logic)
            route_id = \
            routes_df[routes_df['route_short_name'].str.contains(route_name, case=False, na=False)]['route_id'].values[
                0]
            stations_on_route = stop_times_df[
                stop_times_df['trip_id'].isin(trips_df[trips_df['route_id'] == route_id]['trip_id'].tolist())][
                'stop_id'].unique()
            station_list = ", ".join(stops_df[stops_df['stop_id'].isin(stations_on_route)]['stop_name'].tolist())
            dispatcher.utter_message(text=f"Stations on route {route_name}: {station_list}")

        return []


class ActionRealTimeInformation(Action):

    def name(self) -> str:
        return "action_real_time_information"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        station_name = tracker.get_slot('station_name')
        if station_name:
            # Mock logic for real-time information
            real_time_info = f"{station_name} is currently open with no service alerts."
            dispatcher.utter_message(text=real_time_info)
        return []
