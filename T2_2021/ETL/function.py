from io import StringIO
import pandas as pd
import datetime
import boto3
import botocore
from sodapy import Socrata

def get_csv(bucket, key, s3_resource, fallback = None):
    try:
        # try and get the csv from s3 if it exists
        s3_resource.Object(bucket, key).load()
        read_file = s3_resource.Object(bucket, key).get()  #note key is on line 27
        df = pd.read_csv(read_file['Body'])
        return df # return the s3 csv
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return fallback # if the key does not exist then we return the fallback csv
        else:
            raise # throw error because something else is wrong
def run(event, context):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    
    a = datetime.datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz = "UTC")
    d = ts.tz_convert(tz='Australia/Victoria')
    key = f'parkingsensor/daily/{d.date()}.csv'
    
    # get the csv for the key above, if it can't fetch the csv will return the fallback (empty df)
    df = get_csv(
        bucket, 
        key, 
        s3_resource, 
        fallback = pd.DataFrame({'bay_id': [], 'st_marker_id': [], 'status': [], 'datetime': []})
    )
    client = Socrata(
        "data.melbourne.vic.gov.au",
        "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
        timeout=120
    )
    
    # add another number to the csv
    df1 = pd.DataFrame(client.get("vh2v-4nfs", limit=200000))
    df1.drop(columns = ['location', 'lat', 'lon', ':@computed_region_evbi_jbp8'], inplace = True)
 #  df1['datetime'] = datetime.datetime.today().replace(microsecond=0) + datetime.timedelta(hours = 10) #Melbourne Time
    df1['datetime'] = d  #This is changing it to Melbourne Timezone won't be affect by DST
    df = df.append(df1) # append the new data to the dataframe

    # write the csv to a buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index = False)
    s3_resource.Object(bucket, f'parkingsensor/daily/{d.date()}.csv').put(Body=csv_buffer.getvalue())
    return f"CSV now has {len(df)} rows"
