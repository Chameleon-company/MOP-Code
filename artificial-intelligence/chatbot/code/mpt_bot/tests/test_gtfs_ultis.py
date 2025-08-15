import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import spacy
from fuzzywuzzy import process
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from actions.gtfs_utils import GTFSUtils
import requests
import logging

# Setup logger to debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

train_data = GTFSUtils.load_mode_data("mpt_data/2", "train")

if train_data:
    stops_df, stop_times_df, routes_df, trips_df, calendar_df = train_data
else:
    logger.error("Failed to load Train data.")

class TestGTFSUtils(unittest.TestCase):
    def setUp(self):
        # Sample stops.txt data
        self.stops_data = stops_df

        # Sample stop_times.txt data
        self.stop_times_data = stop_times_df

        self.routes_df = routes_df
        self.trips_df = trips_df

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
        self.patcher_requests = patch('requests.get')
        self.mock_requests = self.patcher_requests.start()

    def tearDown(self):
        self.patcher_spacy.stop()
        self.patcher_fuzzy.stop()
        self.patcher_requests.stop()

    # def test_find_station_name_exact_match(self):
    #     """Test find_station_name with exact match."""
    #     self.mock_fuzzy.return_value = [('syndal station', 100), ('flinders street station', 80)]
    #     result = GTFSUtils.find_station_name('Syndal Station', self.stops_data)
    #     self.assertEqual(result, 'Syndal Station')

    # def test_find_station_name_partial_match(self):
    #     """Test find_station_name with partial match and synonym."""
    #     self.mock_fuzzy.return_value = [('melbourne central station', 95), ('flinders street station', 80)]
    #     result = GTFSUtils.find_station_name('melbourne cen', self.stops_data)
    #     self.assertEqual(result, 'Melbourne Central Station')

    # def test_find_station_name_no_match(self):
    #     """Test find_station_name with no valid match."""
    #     self.mock_fuzzy.return_value = [('sunshine station', 70)]
    #     result = GTFSUtils.find_station_name('invalid', self.stops_data)
    #     self.assertIsNone(result)

    # def test_extract_stations_from_query_two_stations(self):
    #     """Test extract_stations_from_query with two stations in query."""
    #     query = "What’s the best way to go from Sunshine to Glenferrie?"
    #     self.mock_fuzzy.side_effect = [
    #         [('sunshine station', 100)],
    #         [('glenferrie station', 95)]
    #     ]
    #     # Remove tracker if function doesn't accept it
    #     result = GTFSUtils.extract_stations_from_query(query, self.stops_data)
    #     self.assertEqual(result, ['Sunshine Station', 'Glenferrie Station'])

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
    #         'Box Hill Station', 'Flinders Street Station', self.stops_data, self.stop_times_data
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

    def test_check_route_and_fetch_disruptions(self):
        """Test check_route_and_fetch_disruptions with mocked PTV API."""
        self.mock_requests.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "disruptions": {
                    "general": [],
                    "metro_train": [
                    {
                        "disruption_id": 341246,
                        "title": "Werribee Line: Buses replace trains from first service Thursday 14 August to 8.30pm Friday 15 August 2025",
                        "url": "http://ptv.vic.gov.au/live-travel-updates/article/werribee-line-buses-replace-trains-from-first-service-to-830pm-each-day-from-thursday-14-august-to-8-30pm-friday-15-august-2025",
                        "description": "Buses replace trains between Laverton and Werribee from first service Thursday 14 August to 8.30pm Friday 15 August, due to level crossing removal works.",
                        "disruption_status": "Current",
                        "disruption_type": "Planned Works",
                        "published_on": "2025-08-11T22:48:51Z",
                        "last_updated": "2025-08-13T17:01:35Z",
                        "from_date": "2025-08-13T17:00:00Z",
                        "to_date": "2025-08-15T10:30:00Z",
                        "routes": [
                        {
                            "route_type": 0,
                            "route_id": 16,
                            "route_name": "Werribee",
                            "route_number": "",
                            "route_gtfs_id": "2-WER",
                            "direction": null
                        }
                        ],
                        "stops": [],
                        "colour": "#ffd500",
                        "display_on_board": false,
                        "display_status": false
                    }]
                }
            }
        )
        
        result = GTFSUtils.check_route_and_fetch_disruptions('Werribee', 'train', self.routes_df)
        print(result)
        self.assertTrue(result)
        #self.assertEqual(result[0]['disruption_id'], 123)
        #self.assertEqual(result[0]['title'], 'Delay on Frankston Line')

    # def test_check_route_and_fetch_disruptions_no_disruptions(self):
    #     """Test check_route_and_fetch_disruptions with no disruptions."""
    #     self.mock_requests.return_value = MagicMock(
    #         status_code=200,
    #         json=lambda: {'disruptions': []}
    #     )
        
    #     result = GTFSUtils.check_route_and_fetch_disruptions('Belgrave Line', 'train', self.routes_df)
    #     self.assertEqual(result, [])



if __name__ == '__main__':
    unittest.main()