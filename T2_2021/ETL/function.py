from io import StringIO
import pandas as pd
import datetime
import boto3
from sodapy import Socrata

def run(event, context):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/parkingsensor.csv'
    read_file = s3_resource.Object(bucket, key).get()

    client = Socrata(
        "data.melbourne.vic.gov.au",
        "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
        timeout=120
    )

    df = pd.read_csv(read_file['Body'])
    # add another number to the csv
    df1 = pd.DataFrame(client.get("vh2v-4nfs", limit=200000))
    df1.drop(columns = ['location', 'lat', 'lon', ':@computed_region_evbi_jbp8'], inplace = True)
    df1['datetime'] = datetime.datetime.today().replace(microsecond=0) + datetime.timedelta(hours = 10) #Melbourne Time
    df = df.append(df1)


    # write the csv to a buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index = False)

    s3_resource.Object(bucket, 'parkingsensor/parkingsensor.csv').put(Body=csv_buffer.getvalue())
    return f"CSV now has {len(df)} rows"
