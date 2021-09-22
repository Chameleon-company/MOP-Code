import boto3
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
import threading

thread_local = threading.local()

def get(bucket, resource):
    
    def get_key(key):
        thread_local.reader = resource.Object(bucket, key).get()
        return thread_local.reader

    return get_key

''' Returns Parking sensor csv'''
def get_parking_sensor_data():
    # df = pd.read_csv('flaskr/parking_sensor/data/parkingsensor.csv', parse_dates=True, infer_datetime_format=True)
    # return df
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')

    df = None
    now = datetime.today().replace(microsecond=0)

    num_days = 28
    keys = [f'parkingsensor/daily/{now.date() - timedelta(days=i)}.csv' for i in range(num_days)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_days) as executor:
        for reader in executor.map(get(bucket, s3_resource), keys):
            day_file = pd.read_csv(reader['Body'])
            df = day_file if df is None else df.append(day_file)

    return df
