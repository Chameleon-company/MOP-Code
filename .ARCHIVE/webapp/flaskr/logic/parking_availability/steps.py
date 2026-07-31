import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import jsonify
from geopy.distance import geodesic
from pandas.core.frame import DataFrame
from sodapy.socrata import Socrata

from .data import get_parking_sensor_data

###### STEP - Access Data ######

'''
  Fetch the latest parking information and return json
'''


def get_live_parking_json():
    parking_dataset_id = 'vh2v-4nfs'

    client = Socrata(
        "data.melbourne.vic.gov.au",
        # app token, just used to reduce throttling, not authentication
        "EC65cHicC3xqFXHHvAUICVXEr",
        timeout=120
    )

    # 003 read current snapshot of parking sensors' status
    api_results = client.get_all(parking_dataset_id)
    parking_sensors = pd.DataFrame.from_dict(api_results)
    parking_sensors = parking_sensors[[
        'bay_id', 'st_marker_id', 'status', 'lat', 'lon']]
    parking_sensors = parking_sensors.astype(
        {'lat': 'float64', 'lon': 'float64'})
    parking_sensors = parking_sensors.rename(
        columns={'st_marker_id': 'marker_id'})
    # remove duplicates found in the parking sensor data
    parking_sensors = parking_sensors.drop_duplicates()
    parking_sensors['status'] = parking_sensors['status'].fillna('Unknown')

    results = parking_sensors[['lat', 'lon', 'status']].to_dict('records')

    return jsonify(results)


##### STEP - Visualize Data ######


''' This function will also fetch the latest parking sensor data, but will perform less reformatting than above
    it also adds datetime.now information to a datetime column
'''


def get_live_parking():
    # find the parking dataset @ https://data.melbourne.vic.gov.au/Transport/On-street-Parking-Bay-Sensors/vh2v-4nfs
    parking_dataset_id = 'vh2v-4nfs'

    client = Socrata(
        "data.melbourne.vic.gov.au",
        # app token, just used to reduce throttling, not authentication
        "EC65cHicC3xqFXHHvAUICVXEr",
        timeout=120
    )

    api_results = client.get_all(parking_dataset_id)
    parking_sensors = pd.DataFrame.from_dict(api_results)
    parking_sensors = parking_sensors[[
        'bay_id', 'st_marker_id', 'status', 'lat', 'lon']]
    parking_sensors = parking_sensors.astype(
        {'lat': 'float64', 'lon': 'float64'})
    parking_sensors = parking_sensors.rename(
        columns={'st_marker_id': 'marker_id'})

    parking_sensors = parking_sensors.drop_duplicates()
    parking_sensors['status'] = parking_sensors['status'].fillna('Unknown')
    a = datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz="UTC")
    d = ts.tz_convert(tz='Australia/Victoria')
    parking_sensors['datetime'] = d

    return parking_sensors


'''
This function will take in a data frame with entries for each sensor with respective day of week, and status columns
and return a dataframe of the form [{'DayOfWeek': string, 'Percentage': float32}]
'''


def get_daily_percentage_availability(df):
    df['DayOfWeek'] = df['datetime'].dt.day_of_week
    DailyParkingCounts = df.groupby('DayOfWeek').status.value_counts()
    DailyParkingCounts = DailyParkingCounts.unstack().reset_index()
    DailyParkingCounts['Percentage'] = 100 * (DailyParkingCounts['Unoccupied'] / (
        DailyParkingCounts['Unoccupied'] + DailyParkingCounts['Present']))
    DailyParkingCounts.reset_index(drop=True)
    return DailyParkingCounts[['DayOfWeek', 'Percentage']]


'''
 Same as above function except deal with hourly availability as opposed to daily
'''


def get_hourly_availability_trend(df):
    df['Hours'] = df['datetime'].dt.hour
    DailyAvailability = df.groupby('Hours').status.value_counts()
    DailyAvailability = DailyAvailability.unstack().reset_index()
    DailyAvailability['Availability'] = 100 * DailyAvailability['Unoccupied'] / \
        (DailyAvailability['Present'] + DailyAvailability['Unoccupied'])
    DailyAvailability = DailyAvailability.reset_index(drop=True)
    return DailyAvailability[['Hours', 'Availability']]


'''
    This function takes in an expected daily trend DataFrame as produced by the function above,
    and also a smaller 'current' DataFrame which has the same schema but with data only covering the current day
    of parking sensor activity.
    
    The function returns a buffer with the bytes of the visualization
'''


def visualize_trend(
        expected,
        current,
        x_column='DayOfWeek',
        y_column='Percentage',
        plot_params={'ylabel': 'Parking Percentage', 'title': 'Parking Availability', 'xlabel': 'Day of the Week'}):

    # Visualize the results
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 6), dpi=80)
    sns.set_style("whitegrid")
    plt.title(plot_params['title'])
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'], labelpad=14)

    plt.bar(expected[x_column], expected[y_column],
            alpha=0.4, label="Expected")
    plt.bar(current[x_column], current[y_column], alpha=0.4, label="Current")

    if 'xticks' in plot_params.keys():
        plt.xticks(ticks=plot_params['xticks'],
                   labels=plot_params['xtick_labels'])

    if 'yticks' in plot_params.keys():
        plt.yticks(ticks=plot_params['yticks'],
                   labels=plot_params['ytick_labels'])

    plt.legend(loc="lower left", borderaxespad=1)
    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visualize_daily_latest():
    # load data from csv file
    df = get_parking_sensor_data()
    # df = pd.read_csv('data/parkingsensor_collection.csv', parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(
        df['datetime'], infer_datetime_format=True, utc=True)
    current_df = get_live_parking()
    # perform analysis
    daily_percentage = get_daily_percentage_availability(df)
    # perform analysis limited to today
    current_daily_percentage = get_daily_percentage_availability(current_df)
    # visualize results
    return visualize_trend(daily_percentage, current_daily_percentage, plot_params={
        'title': 'Daily Parking Availability',
        'xlabel': 'Days',
        'ylabel': '% Availability',
        'xticks': [0, 1, 2, 3, 4, 5, 6],
        'xtick_labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    })


def visualize_hourly_latest():
    df = get_parking_sensor_data()
    df['datetime'] = pd.to_datetime(
        df['datetime'], infer_datetime_format=True, utc=True)

    # subset of data for only todays date
    current_hour_df = get_live_parking()

    expected_hourly = get_hourly_availability_trend(df)
    current_hourly = get_hourly_availability_trend(current_hour_df)
    # visualize results
    return visualize_trend(
        expected_hourly,
        current_hourly,
        'Hours',
        'Availability',
        {
            'ylabel': '% Availability',
            'xlabel': 'Time',
            'title': 'Hourly Parking Availability',
            'xticks': list(range(0, 24, 2)),
            'xtick_labels': [
                '12AM',
                '2AM',
                '4AM',
                '6AM',
                '8AM',
                '10AM',
                '12PM',
                '2PM',
                '4PM',
                '6PM',
                '8PM',
                '10PM']
        })


##### STEP - Geofiltered Visualization of Data ######

def get_filtered_ids(lat, lng, radius):
    # read in the base list of parking sensors
    df_baselist = (DataFrame)(pd.read_csv(
        'flaskr/logic/parking_availability/datasets/ps_baselist.csv'))
    # initialisation of filter loop
    pin = (lat, lng)
    r = int(radius)
    j = 0
    # create
    marker_ids = []
    # filter loop
    for i in np.arange(0, df_baselist.shape[0]):
        d = geodesic(pin, (df_baselist.lati[i], df_baselist.long[i])).meters
        if d <= r:
            marker_ids.append(df_baselist.st_marker_id[i])
            j = j+1
        else:
            continue

    return marker_ids


def visualize_filtered_daily_latest(lat, lng, radius):
    df = get_parking_sensor_data()

    # df = pd.read_csv('data/parkingsensor_collection.csv', parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(
        df['datetime'], infer_datetime_format=True, utc=True)

    marker_ids = get_filtered_ids(lat, lng, radius)

    # df below is result of filter by circle
    df = df[df["st_marker_id"].isin(marker_ids)]

    current_df = get_live_parking()
    # perform analysis
    daily_percentage = get_daily_percentage_availability(df)
    # perform analysis limited to today
    current_daily_percentage = get_daily_percentage_availability(current_df)

    # visualize results
    return visualize_trend(daily_percentage, current_daily_percentage, plot_params={
        'title': 'Daily Parking Availability',
        'xlabel': 'Days',
        'ylabel': '% Availability',
        'xticks': [0, 1, 2, 3, 4, 5, 6],
        'xtick_labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']})


def visualize_filtered_hourly_latest(lat, lng, radius):

    df = get_parking_sensor_data()

    df['datetime'] = pd.to_datetime(
        df['datetime'], infer_datetime_format=True, utc=True)

    marker_ids = get_filtered_ids(lat, lng, radius)

    # df below is result of filter by circle
    df = df[df["st_marker_id"].isin(marker_ids)]

    # subset of data for only todays date
    current_hour_df = get_live_parking()

    expected_hourly = get_hourly_availability_trend(df)
    current_hourly = get_hourly_availability_trend(current_hour_df)

    # visualize results
    return visualize_trend(expected_hourly, current_hourly, 'Hours', 'Availability', plot_params={
        'ylabel': '% Availability',
        'xlabel': 'Time',
        'title': 'Hourly Parking Availability',
        'xticks': list(range(0, 24, 2)),
        'xtick_labels': [
            '12AM',
            '2AM',
            '4AM',
            '6AM',
            '8AM',
            '10AM',
            '12PM',
            '2PM',
            '4PM',
            '6PM',
            '8PM',
            '10PM']
    })
