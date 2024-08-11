import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
import os
import sys
from fuzzywuzzy import process
from gt_parser import load_gtfs_data 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
stops, routes, trips, stop_times = load_gtfs_data()

class ActionNextTrain(Action):

    def name(self) -> str:
        return "action_next_train"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> list:
        
        stop_name = next(tracker.get_latest_entity_values('stop'), None)
        
        if stop_name:
            stop_names = stops['stop_name'].tolist()
            best_match = process.extractOne(stop_name, stop_names)
            if best_match[1] < 80:  # threshold for fuzzy matching
                dispatcher.utter_message(text=f"Sorry, I couldn't find a stop matching {stop_name}.")
                return []
            
            stop_id = stops[stops['stop_name'] == best_match[0]]['stop_id'].values[0]
            next_trains = stop_times[stop_times['stop_id'] == stop_id].sort_values(by='arrival_time')
            if next_trains.empty:
                dispatcher.utter_message(text=f"There are no upcoming trains at {best_match[0]}.")
            else:
                next_train_time = next_trains['arrival_time'].values[0]
                dispatcher.utter_message(text=f"The next train at {best_match[0]} is at {next_train_time}.")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find that stop.")
        
        return []

class ActionNextBus(Action):

    def name(self) -> str:
        return "action_next_bus"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> list:
        
        stop_name = next(tracker.get_latest_entity_values('stop'), None)
        
        if stop_name:
            stop_names = stops['stop_name'].tolist()
            best_match = process.extractOne(stop_name, stop_names)
            if best_match[1] < 80:  # threshold for fuzzy matching
                dispatcher.utter_message(text=f"Sorry, I couldn't find a stop matching {stop_name}.")
                return []
            
            stop_id = stops[stops['stop_name'] == best_match[0]]['stop_id'].values[0]
            next_buses = stop_times[stop_times['stop_id'] == stop_id].sort_values(by='arrival_time')
            if next_buses.empty:
                dispatcher.utter_message(text=f"There are no upcoming buses at {best_match[0]}.")
            else:
                next_bus_time = next_buses['arrival_time'].values[0]
                dispatcher.utter_message(text=f"The next bus at {best_match[0]} is at {next_bus_time}.")
        else:
            dispatcher.utter_message(text="Sorry, I couldn't find that stop.")
        
        return []

class ActionListStops(Action):

    def name(self) -> str:
        return "action_list_stops"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> list:
        
        stop_names = stops['stop_name'].tolist()
        available_stops = ", ".join(stop_names[:10])  # Return first 10 stops for brevity
        dispatcher.utter_message(text=f"Here are some of the available stops: {available_stops}")
        
        return []
