import botocore
import pandas as pd


def get_csv(bucket, key, s3_resource, fallback=None):
    try:
        # try and get the csv from s3 if it exists
        s3_resource.Object(bucket, key).load()
        read_file = s3_resource.Object(
            bucket, key).get()  # note key is on line 27
        df = pd.read_csv(read_file['Body'])
        return df  # return the s3 csv
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return fallback  # if the key does not exist then we return the fallback csv
        else:
            raise  # throw error because something else is wrong
