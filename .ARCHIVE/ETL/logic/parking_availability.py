import datetime
from io import StringIO

import boto3
import pandas as pd
from sodapy import Socrata

from .data import get_csv

""" Given a schedule -
 will update the csv file for the corresponding day with the parking sensor statuses for the given day
"""


def update_daily_parking():
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')

    a = datetime.datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz="UTC")
    d = ts.tz_convert(tz='Australia/Victoria')
    key = f'parkingsensor/daily/{d.date()}.csv'

    # get the csv for the key above, if it can't fetch the csv will return the fallback (empty df)
    df = get_csv(
        bucket,
        key,
        s3_resource,
        fallback=pd.DataFrame(
            {'bay_id': [], 'st_marker_id': [], 'status': [], 'datetime': []})
    )
    client = Socrata(
        "data.melbourne.vic.gov.au",
        # app token, just used to reduce throttling, not authentication
        "EC65cHicC3xqFXHHvAUICVXEr",
        timeout=120
    )

    # add another number to the csv
    df1 = pd.DataFrame(client.get("vh2v-4nfs", limit=200000))
    df1.drop(columns=['location', 'lat', 'lon',
             ':@computed_region_evbi_jbp8'], inplace=True)
 #  df1['datetime'] = datetime.datetime.today().replace(microsecond=0) + datetime.timedelta(hours = 10) #Melbourne Time
    # This is changing it to Melbourne Timezone won't be affect by DST
    df1['datetime'] = d
    df = df.append(df1)  # append the new data to the dataframe

    # write the csv to a buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource.Object(
        bucket, f'parkingsensor/daily/{d.date()}.csv').put(Body=csv_buffer.getvalue())
    return f"CSV now has {len(df)} rows"


def collect_parkingsensor():
    import datetime as dt
    current_time = dt.datetime.now()

    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')

    def get_daily_csv(filename): return get_csv(
        bucket,
        filename,
        s3_resource,
        fallback=pd.DataFrame(
            {'bay_id': [], 'st_marker_id': [], 'status': [], 'datetime': []})
    )

    # collect data from previous 29 days
    dataframe = None
    for day in range(1, 29):
        d = current_time - dt.timedelta(days=day)
        daily_filename = f'parkingsensor/daily/{d.date()}.csv'
        csv = get_daily_csv(
            daily_filename)

        if dataframe is None:
            dataframe = csv
        else:
            dataframe = dataframe.append(csv)

    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, index=False)
    s3_resource.Object(
        bucket, f'parkingsensor/parkingsensor.csv').put(Body=csv_buffer.getvalue())
