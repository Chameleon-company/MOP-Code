from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import pandas as pd
import os

# Load GTFS data from selected folders
def load_gtfs_data(folder_paths):
    data = {}
    for name, folder in folder_paths.items():
        folder_path = os.path.join(folder, 'google_transit')
        data[name] = {
            "stops": pd.read_csv(os.path.join(folder_path, 'stops.txt')),
            "routes": pd.read_csv(os.path.join(folder_path, 'routes.txt')),
            "trips": pd.read_csv(os.path.join(folder_path, 'trips.txt')),
            "stop_times": pd.read_csv(os.path.join(folder_path, 'stop_times.txt'))
        }
    return data

folders = {
    "regional_train": os.path.join('data', 'gtfs', '1'),
    "metropolitan_train": os.path.join('data', 'gtfs', '2')
}

gtfs_data = load_gtfs_data(folders)

class ActionFindNextTransport(Action):

    def name(self) -> str:
        return "action_find_next_transport"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        stop_name = tracker.get_slot('station_name')
        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')

        dispatcher.utter_message(text=f"Received stop_name: {stop_name}, transport_type: {transport_type}")

        if not stop_name or not transport_type:
            dispatcher.utter_message(text="Please provide a stop name and transport type.")
            return []

        data = gtfs_data.get(transport_type)
        if not data:
            dispatcher.utter_message(text="Invalid transport type provided.")
            return []

        try:
            # Find stop ID
            stop_id = data['stops'][data['stops']['stop_name'].str.contains(stop_name, case=False, na=False)]['stop_id'].values[0]

            # Find next trip at this stop
            next_trip = data['stop_times'][data['stop_times']['stop_id'] == stop_id].sort_values(by='departure_time').iloc[0]

            trip_id = next_trip['trip_id']
            route_id = data['trips'][data['trips']['trip_id'] == trip_id]['route_id'].values[0]
            route_name = data['routes'][data['routes']['route_id'] == route_id]['route_long_name'].values[0]
            departure_time = next_trip['departure_time']

            message = f"The next {transport_type.replace('_', ' ')} at {stop_name} is route {route_name} at {departure_time}."
            dispatcher.utter_message(text=message)
        except IndexError:
            dispatcher.utter_message(text=f"Sorry, I couldn't find any upcoming {transport_type.replace('_', ' ')} for {stop_name}.")
        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")

        return []

class ActionFindNearbyStation(Action):

    def name(self) -> str:
        return "action_find_nearby_station"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        dispatcher.utter_message(text=f"Received transport_type: {transport_type}")

        if not transport_type:
            dispatcher.utter_message(text="Please specify the transport type.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        nearby_station = gtfs_data[transport_type]["stops"].iloc[0]['stop_name']
        dispatcher.utter_message(text=f"The nearest station for {transport_type.replace('_', ' ')} is {nearby_station}.")
        return []

class ActionSearchByName(Action):

    def name(self) -> str:
        return "action_search_by_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        station_name = tracker.get_slot('station_name')

        dispatcher.utter_message(text=f"Received station_name: {station_name}, transport_type: {transport_type}")

        if not transport_type or not station_name:
            dispatcher.utter_message(text="Please provide both transport type and station name.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        station_info = gtfs_data[transport_type]["stops"][gtfs_data[transport_type]["stops"]['stop_name'].str.contains(station_name, case=False, na=False)]
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

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        stop_id = tracker.get_slot('stop_id')

        dispatcher.utter_message(text=f"Received stop_id: {stop_id}, transport_type: {transport_type}")

        if not transport_type or not stop_id:
            dispatcher.utter_message(text="Please provide both transport type and stop ID.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        stop_info = gtfs_data[transport_type]["stops"][gtfs_data[transport_type]["stops"]['stop_id'] == int(stop_id)]
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

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        station_name = tracker.get_slot('station_name')

        dispatcher.utter_message(text=f"Received station_name: {station_name}, transport_type: {transport_type}")

        if not transport_type or not station_name:
            dispatcher.utter_message(text="Please provide both transport type and station name.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        station_info = gtfs_data[transport_type]["stops"][gtfs_data[transport_type]["stops"]['stop_name'].str.contains(station_name, case=False, na=False)]
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

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        station_name = tracker.get_slot('station_name')

        dispatcher.utter_message(text=f"Received station_name: {station_name}, transport_type: {transport_type}")

        if not transport_type or not station_name:
            dispatcher.utter_message(text="Please provide both transport type and station name.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

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

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        zone = tracker.get_slot('zone')
        route_name = tracker.get_slot('route_name')

        dispatcher.utter_message(text=f"Received zone: {zone}, route_name: {route_name}, transport_type: {transport_type}")

        if not transport_type:
            dispatcher.utter_message(text="Please provide the transport type.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        if zone:
            stations_in_zone = gtfs_data[transport_type]["stops"][gtfs_data[transport_type]["stops"]['stop_name'].str.contains(f'Zone {zone}', case=False, na=False)]
            station_list = ", ".join(stations_in_zone['stop_name'].tolist())
            dispatcher.utter_message(text=f"Stations in Zone {zone}: {station_list}")

        elif route_name:
            # Assuming we have a route-station mapping (simplified logic)
            route_id = gtfs_data[transport_type]["routes"][gtfs_data[transport_type]["routes"]['route_short_name'].str.contains(route_name, case=False, na=False)]['route_id'].values[0]
            stations_on_route = gtfs_data[transport_type]["stop_times"][gtfs_data[transport_type]["stop_times"]['trip_id'].isin(gtfs_data[transport_type]["trips"][gtfs_data[transport_type]["trips"]['route_id'] == route_id]['trip_id'].tolist())]['stop_id'].unique()
            station_list = ", ".join(gtfs_data[transport_type]["stops"][gtfs_data[transport_type]["stops"]['stop_id'].isin(stations_on_route)]['stop_name'].tolist())
            dispatcher.utter_message(text=f"Stations on route {route_name}: {station_list}")

        return []

class ActionRealTimeInformation(Action):

    def name(self) -> str:
        return "action_real_time_information"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        transport_type = tracker.get_slot('transport_type').lower().replace(' ', '_')
        station_name = tracker.get_slot('station_name')

        dispatcher.utter_message(text=f"Received station_name: {station_name}, transport_type: {transport_type}")

        if not transport_type or not station_name:
            dispatcher.utter_message(text="Please provide both transport type and station name.")
            return []

        if transport_type not in gtfs_data:
            dispatcher.utter_message(text=f"Transport type {transport_type} is not recognized.")
            return []

        # Mock logic for real-time information
        real_time_info = f"{station_name} is currently open with no service alerts."
        dispatcher.utter_message(text=real_time_info)
        return []
