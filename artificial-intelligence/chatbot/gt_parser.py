import os
import pandas as pd

def load_gtfs_data(gtfs_base_path='gtfs'):
    stops_list = []
    routes_list = []
    trips_list = []
    stop_times_list = []

    # Iterate through each folder in the GTFS base path
    for folder_name in os.listdir(gtfs_base_path):
        folder_path = os.path.join(gtfs_base_path, folder_name, 'google_transit')
        if os.path.isdir(folder_path):
            stops_path = os.path.join(folder_path, 'stops.txt')
            routes_path = os.path.join(folder_path, 'routes.txt')
            trips_path = os.path.join(folder_path, 'trips.txt')
            stop_times_path = os.path.join(folder_path, 'stop_times.txt')
            
            if os.path.exists(stops_path):
                stops_list.append(pd.read_csv(stops_path))
            if os.path.exists(routes_path):
                routes_list.append(pd.read_csv(routes_path))
            if os.path.exists(trips_path):
                trips_list.append(pd.read_csv(trips_path))
            if os.path.exists(stop_times_path):
                stop_times_list.append(pd.read_csv(stop_times_path))

    # Concatenate all dataframes
    stops = pd.concat(stops_list, ignore_index=True)
    routes = pd.concat(routes_list, ignore_index=True)
    trips = pd.concat(trips_list, ignore_index=True)
    stop_times = pd.concat(stop_times_list, ignore_index=True)

    # Print first few rows of stops to verify data
    print("First few stops:", stops.head())

    return stops, routes, trips, stop_times

# Example usage (uncomment to test)
# stops, routes, trips, stop_times = load_gtfs_data()
