import boto3
from sodapy.socrata import Socrata
import geopandas as gpd
import pandas as pd
from flask import jsonify
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic

###### STEP - Access Data ######



'''
  Fetch the latest parking information and return json
'''
def get_live_parking_json():
    # start_index = request.args['start'] if request.args['stop'] is not None else 0
    # stop_index = request.args['stop'] if request.args['stop'] is not None else 100

    # find the parking dataset @ https://data.melbourne.vic.gov.au/Transport/On-street-Parking-Bay-Sensors/vh2v-4nfs
    parking_dataset_id = 'vh2v-4nfs'
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'static_datasets/006_crvt-b4kt_dl_at__20210808_manual.geojson'
    # read_file = s3_resource.Object(bucket, key).get()

    client = Socrata(
        "data.melbourne.vic.gov.au",
        "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
        timeout=120
    )

    # historic_df = gpd.read_file(read_file['Body'])
    # # use the centroid of the polygon to estimate the sensor location
    # historic_df['centroid'] = historic_df.centroid
    # historic_df['lati'] = historic_df['centroid'].y
    # historic_df['long'] = historic_df['centroid'].x
    # historic_df = historic_df.drop(columns=['bay_id','meter_id','last_edit'])
    # historic_df = historic_df.fillna(value=np.nan)
    # historic_df = historic_df[historic_df['marker_id'].notna()]
    # historic_df = historic_df.drop_duplicates('marker_id')

    ## 003 read current snapshot of parking sensors' status
    api_results = client.get_all(parking_dataset_id)
    parking_sensors = pd.DataFrame.from_dict(api_results)
    parking_sensors = parking_sensors[['bay_id','st_marker_id','status','lat','lon']]
    parking_sensors = parking_sensors.astype({'lat':'float64', 'lon':'float64'})
    parking_sensors = parking_sensors.rename(columns={'st_marker_id':'marker_id'})
    # remove duplicates found in the parking sensor data    
    parking_sensors = parking_sensors.drop_duplicates()

    # merge with historically available parking sensors
    # parking_sensors = historic_df.merge(parking_sensors, how='outer', on="marker_id")
    # parking_sensors['lati'] = parking_sensors['lati'].fillna(parking_sensors['lat'])
    # parking_sensors['long'] = parking_sensors['long'].fillna(parking_sensors['lon'])
    # parking_sensors = parking_sensors.fillna(value=np.nan)
    # parking_sensors['lat'] = parking_sensors['lati']
    # parking_sensors['lon'] = parking_sensors['long']
    parking_sensors['status'] = parking_sensors['status'].fillna('Unknown')
    # parking_sensors = parking_sensors.drop(columns=['centroid', 'lati', 'long'])
    
    results = parking_sensors[['lat', 'lon', 'status']].to_dict('records')
    
    return jsonify(results)
    # results = [{'lat': lat,'lng': lng, 'status': status} for (lat, lng), status in zip(zip(results['lat'], results['lng']), results['status'])]



##### STEP - Visualize Data ######

import io


''' This function will also fetch the latest parking sensor data, but will perform less reformatting than above
    it also adds datetime.now information to a datetime column
'''
def get_live_parking():
    # find the parking dataset @ https://data.melbourne.vic.gov.au/Transport/On-street-Parking-Bay-Sensors/vh2v-4nfs
    parking_dataset_id = 'vh2v-4nfs'

    client = Socrata(
        "data.melbourne.vic.gov.au",
        "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
        timeout=120
    )

    api_results = client.get_all(parking_dataset_id)
    parking_sensors = pd.DataFrame.from_dict(api_results)
    parking_sensors = parking_sensors[['bay_id','st_marker_id','status','lat','lon']]
    parking_sensors = parking_sensors.astype({'lat':'float64', 'lon':'float64'})
    parking_sensors = parking_sensors.rename(columns={'st_marker_id':'marker_id'})

    parking_sensors = parking_sensors.drop_duplicates()
    parking_sensors['status'] = parking_sensors['status'].fillna('Unknown')
    a = datetime.datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz = "UTC")
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
    DailyParkingCounts['Percentage'] = (DailyParkingCounts['Unoccupied'] / (DailyParkingCounts['Unoccupied'] + DailyParkingCounts['Present']))
    DailyParkingCounts.reset_index(drop=True)
    return DailyParkingCounts[['DayOfWeek', 'Percentage']]

'''
 Same as above function except deal with hourly availability as opposed to daily
'''
def get_hourly_availability_trend(df):
    df['Hours'] = df['datetime'].dt.hour
    DailyAvailability = df.groupby('Hours').status.value_counts()
    DailyAvailability = DailyAvailability.unstack().reset_index()
    DailyAvailability['Availability'] = DailyAvailability['Unoccupied'] / (DailyAvailability['Present'] + DailyAvailability['Unoccupied'])
    DailyAvailability = DailyAvailability.reset_index(drop=True)
    return DailyAvailability[['Hours', 'Availability']]

'''
    This function takes in an expected daily trend DataFrame as produced by the function above,
    and also a smaller 'current' DataFrame which has the same schema but with data only covering the current day
    of parking sensor activity.
    
    The function returns a buffer with the bytes of the visualization
'''
def visualize_trend(expected, current, x_column='DayOfWeek', y_column = 'Percentage'):
    # Visualize the results
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 6), dpi=80)
    sns.set_style("whitegrid")

    plt.ylabel("Parking Availability", labelpad=14)
    plt.title(f"Parking Availability on {datetime.datetime.today().weekday()} vs Other Days", y=1)
    plt.xlabel("Day of the Week")


    plt.bar(expected[x_column], expected[y_column], alpha=0.4 , label="Expected")
    plt.bar(current[x_column], current[y_column], alpha=0.4 , label="Current")
    plt.xticks({0 : "Monday",
              1: "Tuesday",
              2: "Wednesday",
              3: "Thursday",
              4: "Friday",
              5: "Saturday",
              6: "Sunday"})
    # plt.bar(WednesdayCount['Day_Of_Week'], WednesdayCount['Parking_Availabilities'],alpha=0.4, label="Available Now")
    plt.legend(loc ="lower left", borderaxespad=1)
    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def visualize_daily_latest():
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/parkingsensor.csv'
    read_file = s3_resource.Object(bucket, key).get()

    # load data from csv file
    df = pd.read_csv(read_file['Body'], parse_dates=True, infer_datetime_format=True)
    # df = pd.read_csv('data/parkingsensor_collection.csv', parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, utc=True)
    current_df = get_live_parking()
    # perform analysis
    daily_percentage = get_daily_percentage_availability(df)
    # perform analysis limited to today
    current_daily_percentage = get_daily_percentage_availability(current_df)
    #visualize results
    return visualize_trend(daily_percentage, current_daily_percentage)


def visualize_hourly_latest():
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/parkingsensor.csv'
    read_file = s3_resource.Object(bucket, key).get()

    # load data from csv file
    df = pd.read_csv(read_file['Body'], parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, utc=True)

    # subset of data for only todays date
    current_hour_df = get_live_parking()

    expected_hourly = get_hourly_availability_trend(df)
    current_hourly = get_hourly_availability_trend(current_hour_df)
    #visualize results
    return visualize_trend(expected_hourly, current_hourly, 'Hours', 'Availability')


##### STEP - Geofiltered Visualization of Data ######

def visualize_filtered_daily_latest(lat, lng, radius):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/parkingsensor.csv'
    read_file = s3_resource.Object(bucket, key).get()

    # load data from csv file
    df = pd.read_csv(read_file['Body'], parse_dates=True, infer_datetime_format=True)
    # df = pd.read_csv('flaskr/parking_sensor/data/parkingsensor.csv', parse_dates=True, infer_datetime_format=True)
    # df = pd.read_csv('data/parkingsensor_collection.csv', parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, utc=True)

    ### geo filtering based on lat,lng, radius parameters ###
    # lat, lng are floats, radius is string integer
    # read in base list of 5895 parking bays with marker id's and lati long
    df_baselist = pd.read_csv('flaskr/parking_sensor/data/ps_baselist.csv')
    # initialisation of filter loop
    pin = (lat,lng)
    r = int(radius)
    j = 0
    lst_marker_ids = []
    # filter loop
    for i in np.arange(0,df_baselist.shape[0]):
        d =  geodesic(pin, (df_baselist.lati[i], df_baselist.long[i])).meters
        if d<=r:
            lst_marker_ids.append(df_baselist.st_marker_id[i])
            j=j+1
        else:
            continue
    # df below is result of filter by circle
    df = df[df["st_marker_id"].isin(lst_marker_ids)] 

    current_df = get_live_parking()
    # perform analysis
    daily_percentage = get_daily_percentage_availability(df)
    # perform analysis limited to today
    current_daily_percentage = get_daily_percentage_availability(current_df)
    #visualize results
    return visualize_trend(daily_percentage, current_daily_percentage)

def visualize_filtered_hourly_latest(lat, lng, radius):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/parkingsensor.csv'
    read_file = s3_resource.Object(bucket, key).get()

    # load data from csv file
    df = pd.read_csv(read_file['Body'], parse_dates=True, infer_datetime_format=True)
    # df = pd.read_csv('flaskr/parking_sensor/data/parkingsensor.csv', parse_dates=True, infer_datetime_format=True)
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, utc=True)

    ### geo filtering based on lat,lng, radius parameters ###
    # lat, lng are floats, radius is string integer
    # read in base list of 5895 parking bays with marker id's and lati long
    df_baselist = pd.read_csv('flaskr/parking_sensor/data/ps_baselist.csv')
    # initialisation of filter loop
    pin = (lat,lng)
    r = int(radius)
    j = 0
    lst_marker_ids = []
    # filter loop
    for i in np.arange(0,df_baselist.shape[0]):
        d =  geodesic(pin, (df_baselist.lati[i], df_baselist.long[i])).meters
        if d<=r:
            lst_marker_ids.append(df_baselist.st_marker_id[i])
            j=j+1
        else:
            continue
    # df below is result of filter by circle
    df = df[df["st_marker_id"].isin(lst_marker_ids)] 

    # subset of data for only todays date
    current_hour_df = get_live_parking()

    expected_hourly = get_hourly_availability_trend(df)
    current_hourly = get_hourly_availability_trend(current_hour_df)
    #visualize results
    return visualize_trend(expected_hourly, current_hourly, 'Hours', 'Availability')