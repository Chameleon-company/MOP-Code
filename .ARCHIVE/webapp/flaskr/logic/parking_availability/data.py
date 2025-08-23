import boto3
import pandas as pd


def get():
    bucket = 'opendataplayground.deakin'
    s3_client = boto3.client('s3', region_name="ap-southeast-2")

    def getter(key):
        return s3_client.get_object(Bucket=bucket, Key=key)

    return getter


''' Returns Parking sensor csv'''


def get_parking_sensor_data():
    # df = pd.read_csv('flaskr/parking_sensor/data/parkingsensor.csv', parse_dates=True, infer_datetime_format=True)
    # return df

    df = None
    # now = datetime.today().replace(microsecond=0)

    # num_days = 28
    # keys = [f'parkingsensor/daily/{now.date() - timedelta(days=i)}.csv' for i in range(num_days)]
    # # with concurrent.futures.ThreadPoolExecutor(max_workers=num_days) as executor:
    # getter = get()
    # for key in keys:
    #     reader = getter(key)
    #     day_file = pd.read_csv(reader['Body'])
    #     df = day_file if df is None else df.append(day_file)
    getter = get()
    reader = getter('parkingsensor/parkingsensor.csv')
    df = pd.read_csv(reader['Body'], parse_dates=True,
                     infer_datetime_format=True)

    return df
