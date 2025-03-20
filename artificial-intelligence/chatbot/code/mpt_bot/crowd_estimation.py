import requests
import datetime

def get_crowd_estimate(route_id, stop_id, time_of_day=None):
    """
    Fetch estimated crowd level for a given route and stop.
    :param route_id: ID of the transport route (bus, tram, train)
    :param stop_id: ID of the stop/station
    :param time_of_day: Optional time for estimation, default is current time
    :return: Estimated crowd level (low, medium, high)
    """
    if time_of_day is None:
        time_of_day = datetime.datetime.now().strftime("%H:%M")
    
    # Simulating API request to crowd estimation service
    url = f"https://api.crowddata.com/estimate?route={route_id}&stop={stop_id}&time={time_of_day}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("crowd_level", "unknown")
    else:
        return "Data unavailable"

if __name__ == "__main__":
    route = input("Enter route ID: ")
    stop = input("Enter stop ID: ")
    crowd_level = get_crowd_estimate(route, stop)
    print(f"Estimated crowd level: {crowd_level}")
