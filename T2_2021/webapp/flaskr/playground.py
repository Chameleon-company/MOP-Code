import functools
import json
import numpy as np
from sodapy import Socrata
import boto3
import pandas as pd
import geopandas as gpd
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)
bp = Blueprint('playground', __name__, url_prefix='/playground')

@bp.route("/", methods=('GET',))
def home():
    return render_template('playground/playground.html')

@bp.route("/traffic_lights", methods=('GET',))
def get_traffic_lights():
    df = pd.read_csv('./data/traffic_lights.csv')
    locations = [[lng, lat] for lng, lat in zip(df['Longitude'], df['Latitude'])]
    return jsonify(locations)

@bp.route("/parking-sensors/now", methods=('GET',))
def get_parking_sensor_latest():

    # start_index = request.args['start'] if request.args['stop'] is not None else 0
    # stop_index = request.args['stop'] if request.args['stop'] is not None else 100

    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'static_datasets/006_crvt-b4kt_dl_at__20210808_manual.geojson'
    read_file = s3_resource.Object(bucket, key).get()

    client = Socrata(
        "data.melbourne.vic.gov.au",
        "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
        timeout=120
    )

    historic_df = gpd.read_file(read_file['Body'])
    historic_df['last_edit'] = pd.to_datetime(historic_df['last_edit'])
    historic_df = historic_df.sort_values(by='last_edit', ascending=False)
    historic_df['centroid'] = historic_df.centroid
    historic_df['lati'] = historic_df['centroid'].y
    historic_df['long'] = historic_df['centroid'].x
    historic_df = historic_df.drop(columns=['bay_id','meter_id','last_edit'])
    historic_df = historic_df.fillna(value=np.nan)
    historic_df = historic_df[historic_df['marker_id'].notna()]
    historic_df = historic_df.drop_duplicates('marker_id')

    ## 003 read current snapshot of parking sensors' status
    parking_sensors_dict = client.get_all('vh2v-4nfs')
    parking_sensor_df = pd.DataFrame.from_dict(parking_sensors_dict)
    parking_sensor_df = parking_sensor_df[['bay_id','st_marker_id','status','lat','lon']]
    parking_sensor_df = parking_sensor_df.astype({'lat':'float64', 'lon':'float64'})
    parking_sensor_df = parking_sensor_df.rename(columns={'st_marker_id':'marker_id'})
    # remove duplicates found in the parking sensor data    
    parking_sensor_df = parking_sensor_df.drop_duplicates()

    merged_df = historic_df.merge(parking_sensor_df, how='outer', on="marker_id")
    merged_df['lati'] = merged_df['lati'].fillna(merged_df['lat'])
    merged_df['long'] = merged_df['long'].fillna(merged_df['lon'])
    merged_df = merged_df.fillna(value=np.nan)
    merged_df['lat'] = merged_df['lati']
    merged_df['lng'] = merged_df['long']
    merged_df['status'] = merged_df['status'].fillna('Unknown')
    merged_df = merged_df.drop(columns=['centroid', 'lati', 'long', 'lon'])
    
    results = merged_df.iloc[0:4000]
    results = [{'lat': lat,'lng': lng, 'status': status} for (lat, lng), status in zip(zip(results['lat'], results['lng']), results['status'])]
    return jsonify(results)

@bp.route("/query_location", methods=('GET',))
def query_location():
    lng = request.args["lng"]
    lat = request.args["lat"]

    # do some location query

    return jsonify((lng,lat))


