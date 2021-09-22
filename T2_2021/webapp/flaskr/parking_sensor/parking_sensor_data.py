import boto3
import pandas as pd

''' Returns Parking sensor csv'''
def get_parking_sensor_data():
    df = pd.read_csv('flaskr/parking_sensor/data/parkingsensor.csv', parse_dates=True, infer_datetime_format=True)
    return df
    # bucket = 'opendataplayground.deakin'
    # # get existing dataframe from csv on S3
    # s3_resource = boto3.resource('s3')
    # key = 'parkingsensor/parkingsensor.csv'
    # read_file = s3_resource.Object(bucket, key).get()

    # # load data from csv file
    # return pd.read_csv(read_file['Body'], parse_dates=True, infer_datetime_format=True)
