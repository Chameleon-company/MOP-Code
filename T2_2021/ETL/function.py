from io import StringIO
import pandas as pd
import boto3

def run(event, context):
    bucket = 'opendataplayground.deakin'
    # get existing dataframe from csv on S3
    s3_resource = boto3.resource('s3')
    key = 'parkingsensor/example.csv'
    read_file = s3_resource.Object(bucket, key).get()

    df = pd.read_csv(read_file['Body'])
    # add another number to the csv
    value = df.max().values
    value += 1
    df = df.append([value])

    # write the csv to a buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)

    s3_resource.Object(bucket, 'parkingsensor/example.csv').put(Body=csv_buffer.getvalue())
    return f"CSV now has {len(df)} rows"