

'''
  Fetch the latest parking information and return json
'''
import boto3
from sodapy.socrata import Socrata
import geopandas as gpd
import pandas as pd
from flask import jsonify
import numpy as np

def get_live_parking_json():
    # start_index = request.args['start'] if request.args['stop'] is not None else 0
    # stop_index = request.args['stop'] if request.args['stop'] is not None else 100

    # find the parking dataset @ https://data.melbourne.vic.gov.au/Transport/On-street-Parking-Bay-Sensors/vh2v-4nfs
    parking_dataset_id = 'vh2v-4nfs'
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
    # use the centroid of the polygon to estimate the sensor location
    historic_df['centroid'] = historic_df.centroid
    historic_df['lati'] = historic_df['centroid'].y
    historic_df['long'] = historic_df['centroid'].x
    historic_df = historic_df.drop(columns=['bay_id','meter_id','last_edit'])
    historic_df = historic_df.fillna(value=np.nan)
    historic_df = historic_df[historic_df['marker_id'].notna()]
    historic_df = historic_df.drop_duplicates('marker_id')

    ## 003 read current snapshot of parking sensors' status
    api_results = client.get_all(parking_dataset_id)
    parking_sensors = pd.DataFrame.from_dict(api_results)
    parking_sensors = parking_sensors[['bay_id','st_marker_id','status','lat','lon']]
    parking_sensors = parking_sensors.astype({'lat':'float64', 'lon':'float64'})
    parking_sensors = parking_sensors.rename(columns={'st_marker_id':'marker_id'})
    # remove duplicates found in the parking sensor data    
    parking_sensors = parking_sensors.drop_duplicates()

    # merge with historically available parking sensors
    parking_sensors = historic_df.merge(parking_sensors, how='outer', on="marker_id")
    parking_sensors['lati'] = parking_sensors['lati'].fillna(parking_sensors['lat'])
    parking_sensors['long'] = parking_sensors['long'].fillna(parking_sensors['lon'])
    parking_sensors = parking_sensors.fillna(value=np.nan)
    parking_sensors['lat'] = parking_sensors['lati']
    parking_sensors['lon'] = parking_sensors['long']
    parking_sensors['status'] = parking_sensors['status'].fillna('Unknown')
    parking_sensors = parking_sensors.drop(columns=['centroid', 'lati', 'long'])
    
    results = parking_sensors[['lat', 'lon', 'status']].to_dict('records')
    
    return jsonify(results)
    # results = [{'lat': lat,'lng': lng, 'status': status} for (lat, lng), status in zip(zip(results['lat'], results['lng']), results['status'])]
