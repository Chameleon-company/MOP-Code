from io import StringIO
import pandas as pd
import datetime
import boto3
from sodapy import Socrata

def run(event, context):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    a = datetime.datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz = "UTC")
    d = ts.tz_convert(tz='Australia/Victoria')
    key = f'parkingsensor/{d.date()}.csv'
    
    if read_file = s3_resource.Object(bucket, key).get() == False:  #If file doesn't exist create a new file. PLEASE CHECK
        empty = pd.DataFrame({'bay_id': [], 'st_marker_id': [], 'status': [], 'datetime': []})
        empty.to_csv(f'{d.date()}.csv', index = False)   
    
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
 #  df1['datetime'] = datetime.datetime.today().replace(microsecond=0) + datetime.timedelta(hours = 10) #Melbourne Time
    a = datetime.datetime.today().replace(microsecond=0)
    ts = pd.Timestamp(a, tz = "UTC")
    df1['datetime'] = ts.tz_convert(tz='Australia/Victoria')  #This is changing it to Melbourne Timezone won't be affect by DST
    df = df.append(df1)


    # write the csv to a buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index = False)

    s3_resource.Object(bucket, f'parkingsensor/{d.date()}.csv').put(Body=csv_buffer.getvalue())
    return f"CSV now has {len(df)} rows"


