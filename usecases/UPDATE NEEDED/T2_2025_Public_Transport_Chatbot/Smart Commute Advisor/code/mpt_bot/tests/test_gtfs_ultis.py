import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import spacy
from datetime import datetime, timedelta
from fuzzywuzzy import process
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.gtfs_utils import GTFSUtils
from actions.actions import fetch_route, get_traffic_details, get_traffic_status, geocode_address
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from tabulate import tabulate
import requests
import logging
import json

# Setup logger to debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

class TestGTFSUtils(unittest.TestCase):
    def setUp(self):
        # Sample stops.txt data
        self.stops_data = stops_df

        # Sample stop_times.txt data
        self.stop_times_data = stop_times_df

        self.routes_df = routes_df
        self.trips_df = trips_df

        self.tram_stops = tram_stops
        self.tram_stop_times = tram_stop_times
        self.tram_routes = tram_routes
        self.tram_trips = tram_trips

        self.bus_stops = bus_stops
        self.bus_stop_times = bus_stop_times
        self.bus_routes = bus_routes
        self.bus_trips = bus_trips

        # Normalize data as required by the functions
        # GTFSUtils.normalise_gtfs_data(self.stops_data, self.stop_times_data)

        # Mock spacy
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('sentencizer')
        self.patcher_spacy = patch('spacy.load', return_value=self.nlp)
        self.patcher_spacy.start()

        # Mock fuzzywuzzy
        self.patcher_fuzzy = patch('fuzzywuzzy.process.extractBests')
        self.mock_fuzzy = self.patcher_fuzzy.start()

        # Mock requests for VicRoads API (if needed)
        #self.patcher_requests = patch('requests.get')
        #self.mock_requests = self.patcher_requests.start()

    def tearDown(self):
        self.patcher_spacy.stop()
        self.patcher_fuzzy.stop()
        #self.patcher_requests.stop()

    # def test_find_station_name_exact_match(self):
    #     """Test find_station_name with exact match."""
    #     query = "What is the best route to travel from box hil station to glen waveley station"
    #     self.mock_fuzzy.return_value = [('Flagstaff station', 90), ('syndall station', 90)]
    #     result = GTFSUtils.find_station_name(query, self.stops_data)
    #     print(result)
        #self.assertEqual(result, ['Box Hill Railway Station', 'St Albans Railway Station (St Albans)'])

    # def test_find_station_name_partial_match(self):
    #     """Test find_station_name with partial match and synonym."""
    #     query = "What is the best route to travel from melbourne cen station to Sunshine station"
    #     self.mock_fuzzy.return_value = [('melbourne central station', 95), ('Sunshine station', 90)]
    #     result = GTFSUtils.find_station_name(query, self.stops_data)
    #     self.assertEqual(result, ['Melbourne Central Railway Station', 'Sunshine Railway Station'])

    # def test_find_station_name_no_match(self):
    #     """Test find_station_name with no valid match."""
    #     self.mock_fuzzy.return_value = [('sunshine station', 70)]
    #     result = GTFSUtils.find_station_name('invalid', self.stops_data)
    #     self.assertIsNone(result)

    # def test_extract_stations_from_query_two_stations_exact_matched(self):
    #     """Test extract_stations_from_query with two stations in query."""
    #     query = "What is the best way to go St Albans - Glenferrie station?"
    #     # Remove tracker if function doesn't accept it
    #     result = GTFSUtils.extract_stations_from_query(query, self.stops_data)
    #     self.assertEqual(result, ['St Albans Railway Station (St Albans)', 'Glenferrie Railway Station'])
    
    # def test_extract_stations_from_query_two_stations_no_exact_matched(self):
    #     """Test extract_stations_from_query with two stations in query."""
    #     query = "How can I travel glenfiere - sindal?"
    #     # Remove tracker if function doesn't accept it
    #     result = GTFSUtils.extract_stations_from_query(query, self.stops_data)
    #     self.assertEqual(result, ['Glenferrie Railway Station', 'Syndal Railway Station'])

    # def test_extract_stations_from_query_spacy_fallback(self):
    #     """Test extract_stations_from_query using spaCy fallback."""
    #     query = "When is the next train from box hill station to Parliament station?"
    #     def mock_nlp(text):
    #         doc = self.nlp(text)
    #         doc.ents = [
    #             MagicMock(text='box hill', label_='LOC'),
    #             MagicMock(text='parliament', label_='LOC')
    #         ]
    #         return doc
    #     self.patcher_spacy.return_value = mock_nlp
    #     self.mock_fuzzy.side_effect = [
    #         [('box hill station', 95)],
    #         [('parliament station', 95)]
    #     ]
    #     result = GTFSUtils.extract_stations_from_query(query, self.stops_data)
    #     self.assertEqual(result, ['Box Hill Station', 'Parliament Station'])

    # def test_check_direct_route_true(self):
    #     """Test check_direct_route for a direct route."""
    #     result, trip_ids = GTFSUtils.check_direct_route(
    #         'Sunshine Station', 'Flinders Street Station', self.stops_data, self.stop_times_data
    #     )
    #     self.assertTrue(result)
    #     self.assertTrue(trip_ids)

    # def test_check_direct_route_false(self):
    #     """Test check_direct_route for no direct route."""
    #     result, trip_ids = GTFSUtils.check_direct_route(
    #         'Sunshine Station', 'Camberwell Station', self.stops_data, self.stop_times_data
    #     )
    #     self.assertFalse(result)
    #     self.assertEqual(trip_ids, [])
    
    # def test_calculate_route_travel_time(self):
    #     """Test calculate_route_travel_time for a direct route."""
    #     start_stop = 'Box Hill Station' 
    #     end_stop = 'Flinders Street Station' 
    #     route_list = [start_stop, end_stop]
    #     result = GTFSUtils.calculate_route_travel_time(route_list, self.stops_data, self.stop_times_data)
    #     # Expected time: 23 minutes (from stop_times.txt)
    #     self.assertAlmostEqual(result, 23.0, places=1)

    # def test_calculate_route_travel_time_no_route(self):
        # """Test calculate_route_travel_time for no valid route."""
        # start_stop = 'Footscray Station'  # Sunshine Station
        # end_stop = 'Box Hill Station'  # Box Hill Station
        # route_list = [start_stop, end_stop]
        # result = GTFSUtils.calculate_route_travel_time(route_list, self.stops_data, self.stop_times_data)
        # self.assertIsNone(result)

    # def test_calculate_transfers_direct(self):
    #     """Test calculate_transfers for a direct route (no transfers)."""
    #     # Mock trips_df and routes_df
    
    #     result = GTFSUtils.calculate_transfers(
    #         '10117',  # Sunshine Station
    #         '12197',   # Melbourne Central Station
    #         self.stops_data,
    #         self.stop_times_data,
    #         self.trips_df,
    #         self.routes_df
    #     )
    #     self.assertEqual(len(result), 0)  # No transfers for direct route

    # def test_calculate_transfers_with_transfer(self):
    #     """Test calculate_transfers for a route requiring a transfer."""
        
    #     # Simulate a transfer scenario (not in sample data, but mock stop_times)
    #     stop_times_with_transfer = pd.DataFrame({
    #         'trip_id': ['1', '1', '2', '2'],
    #         'stop_id': ['10117', '11207', '11207', '10922'],
    #         'arrival_time': ['08:00:00', '08:15:00', '08:20:00', '08:35:00'],
    #         'departure_time': ['08:01:00', '08:16:00', '08:21:00', '08:36:00'],
    #         'stop_sequence': [1, 2, 1, 2]
    #     }).set_index(['stop_id', 'trip_id'])
    #     result = GTFSUtils.calculate_transfers(
    #         '10117',  # Sunshine Station
    #         '10922',  # Melbourne Central Station
    #         self.stops_data,
    #         stop_times_with_transfer,
    #         self.trips_df,
    #         self.routes_df
    #     )
    #     self.assertEqual(len(result), 1)  # One transfer at Camberwell Station
    #     self.assertEqual(result[0]['stop_name'], 'Camberwell Station')

    # def test_find_best_route_with_transfers(self):
    #     """Test find_best_route_with_transfers for a route with transfers."""

    
    #     # Mock stop_times with transfer
    #     stop_times_with_transfer = pd.DataFrame({
    #         'trip_id': ['1', '1', '2', '2'],
    #         'stop_id': ['10117', '11207', '11207', '10922'],
    #         'arrival_time': ['08:00:00', '08:15:00', '08:20:00', '08:35:00'],
    #         'departure_time': ['08:01:00', '08:16:00', '08:21:00', '08:36:00'],
    #         'stop_sequence': [1, 2, 1, 2]
    #     }).set_index(['stop_id', 'trip_id'])
    #     result = GTFSUtils.find_best_route_with_transfers(
    #         '10117',  # Sunshine Station
    #         '10922',  # Melbourne Central Station
    #         self.stops_data,
    #         stop_times_with_transfer,
    #         self.trips_df,
    #         self.routes_df
    #     )
    #     self.assertIn('transfers', result)
    #     self.assertEqual(len(result['transfers']), 1)
    #     self.assertEqual(result['transfers'][0]['stop_name'], 'Camberwell Station')
    #     self.assertAlmostEqual(result['total_time'], 35.0, places=1)  # 08:00 to 08:35

    # def test_extract_bus_route_name_valid_query(self):
    #     """Test extract_route_name with a valid route name in the query."""
    #     query = "Are there any bus disruptions on bus 737?"
    #     # Mock routes_df with known route names from stop_times.txt trips
    #     result = GTFSUtils.extract_route_name(query, self.bus_routes)
    #     self.assertEqual(result, '737')

    # def test_extract_route_name_partial_match(self):
    #     """Test extract_route_name with a partial route name in the query."""
    #     query = "Is tram 19 route having disruptions?"
    #     # Mock fuzzywuzzy to simulate partial match
    #     with patch('fuzzywuzzy.process.extractOne') as mock_extract:
    #         mock_extract.return_value = ('19', 95)  # High score for partial match
    #         result = GTFSUtils.extract_route_name(query, self.tram_routes)
    #     self.assertEqual(result, '19')

    # def test_extract_route_name_invalid_query(self):
    #     """Test extract_route_name with an invalid route name in the query."""
    #     query = "Check tram disruptions for tram route Invalid - Invalid?"

    #     with patch('fuzzywuzzy.process.extractOne') as mock_extract:
    #         mock_extract.return_value = ('werribee', 60)  # Low score for invalid match
    #         result = GTFSUtils.extract_route_name(query, self.routes_df)
    #     self.assertIsNone(result)

    # def test_check_route_and_fetch_disruptions_train(self):
    #     """Test check_route_and_fetch_disruptions with mocked PTV API."""
    #     # Load mocked data from JSON file
    #     with open('tests/data/disruptions.json', 'r') as f:
    #         mock_data = json.load(f)
    #     with patch('requests.get') as mock_get:
    #         mock_response = MagicMock()
    #         mock_response.status_code = 200
    #         mock_response.json.return_value = mock_data
    #         mock_get.return_value = mock_response
    #         disruption_list, route_id, error = GTFSUtils.check_route_and_fetch_disruptions('Werribee', 'train', routes_df)
    #     current_time = datetime.utcnow()
    #     self.assertIsInstance(disruption_list, list)
    #     for d in disruption_list:
    #         self.assertTrue(any (route['route_name'].lower() == 'werribee' for route in d['routes']))
    #         self.assertTrue(datetime.fromisoformat(d["from_date"].replace("Z", "")) <= current_time)
    
    # def test_check_route_and_fetch_disruptions_bus(self):
    #     """Test check_route_and_fetch_disruptions with mocked PTV API."""
    #     # Load mocked data from JSON file
    #     with open('tests/data/disruptions.json', 'r') as f:
    #         mock_data = json.load(f)
    #     with patch('requests.get') as mock_get:
    #         mock_response = MagicMock()
    #         mock_response.status_code = 200
    #         mock_response.json.return_value = mock_data
    #         mock_get.return_value = mock_response
    #         disruption_list, route_id, error = GTFSUtils.check_route_and_fetch_disruptions('506', 'bus', bus_routes)

    #     current_time = datetime.utcnow()

    #     self.assertIsInstance(disruption_list, list)
    #     for d in disruption_list:
    #         self.assertTrue(any (route['route_number'] == '506' for route in d['routes']))
    #         self.assertTrue(datetime.fromisoformat(d["from_date"].replace("Z", "")) <= current_time)
    
    # def test_check_route_and_fetch_disruptions_tram(self):
    #     """Test check_route_and_fetch_disruptions with mocked PTV API."""
    #     # Load mocked data from JSON file
    #     with open('tests/data/disruptions.json', 'r') as f:
    #         mock_data = json.load(f)
    #     with patch('requests.get') as mock_get:
    #         mock_response = MagicMock()
    #         mock_response.status_code = 200
    #         mock_response.json.return_value = mock_data
    #         mock_get.return_value = mock_response
    #         disruption_list, route_id, error = GTFSUtils.check_route_and_fetch_disruptions("1", "tram", self.tram_routes)

    #     current_time = datetime.utcnow()
        
    #     self.assertIsInstance(disruption_list, list)
    #     print(disruption_list[0])
    #     for d in disruption_list:
    #         self.assertTrue(any (route['route_number'] == '1' for route in d['routes']))
    #         self.assertTrue(datetime.fromisoformat(d["from_date"].replace("Z", "")) <= current_time)


    # def test_check_route_and_fetch_disruptions_no_disruptions(self):
    #     """Test check_route_and_fetch_disruptions with no disruptions."""
    #     with open('tests/data/disruptions.json', 'r') as f:
    #         mock_data = json.load(f)
    #     with patch('requests.get') as mock_get:
    #         mock_response = MagicMock()
    #         mock_response.status_code = 200
    #         mock_response.json.return_value = mock_data
    #         mock_get.return_value = mock_response
    #         disruption_list, route_id, error = GTFSUtils.check_route_and_fetch_disruptions('Flagstaff', 'train', routes_df)

    #     current_time = datetime.utcnow()
    #     self.assertIsNone(disruption_list)
    # def test_get_station_id(self):
    #     result = GTFSUtils.get_station_id("Syndal Railway Station", self.stops_data)
    #     self.assertEqual(result, "vic:rail:SYN")
    
    # def test_find_child_station(self):
    #     result = GTFSUtils.find_child_station("vic:rail:BOX", self.stops_data)
    #     print(result)

    # def test_geocode_address(self): # works!
    #     address = "Flinders Street station, Melbourne"
    #     data = geocode_address(address)
    #     print(data)

    # def test_fetch_route(self):
    #     source = "Flinders Street station, Melbourne"
    #     destination = "Sunshine Station, Melbourne"
    #     source_coords = geocode_address(source)
    #     destination_coords = geocode_address(destination)
    #     # source_coords = ["-37.84186584","145.26817867"]
    #     # destination_coords = ["-37.82255502","145.04605538"]
        
    #     api_key = "cBr8gVX64Y3q18xGfTKnxHvsGl7AcnMw"
    #     #traffic_data = get_traffic_details(api_key, source_coords)
    #     #route_data = fetch_route(source_coords, destination_coords, api_key)
    #     print(route_data)

    def test_find_all_nearby_stops(self):
        # stop_name = "Albert St/Gisborne St #11"
        # Calculate distance to each stop
        geolocator = Nominatim(user_agent="andre")
        location = geolocator.geocode("Oak Valley South Australia")

        address = f"{location.latitude},{location.longitude}"
        print(f"Address coordinate: {address}")
        nearby_stops, message = GTFSUtils.find_all_nearby_stops(address, "train", self.stops_data)
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

            print(table)
        print(message)

        #departure_list, route_id, error = GTFSUtils.fetch_departures_by_stop(stop_name, "tram", self.tram_stops)
        # print(departure_list[0])
        # self.assertIsInstance(departure_list, list)
        




if __name__ == '__main__':
    unittest.main()