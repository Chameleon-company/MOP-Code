import boto3

"""
 The template for the parking-availability use case requires
 some values to be sent into the template as view model properties.
 This function creates that view model
"""
def build_view_model():
    s3_client = boto3.client('s3')

    # get a public link for the parking_sensor.csv
    parking_sensor_collection_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': 'opendataplayground.deakin',
            'Key': 'parkingsensor/parkingsensor.csv'
        },
        ExpiresIn=3600  # 60 minutes
    )

    # get a public link for the parking_sensor.csv
    parking_sensor_list_url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': 'opendataplayground.deakin',
            'Key': 'parkingsensor/parking_sensors_list.csv'
        },
        ExpiresIn=3600  # 60 minutes
    )

    return {
        'parking_sensor_collection': parking_sensor_collection_url, 
        'parking_sensors_list': parking_sensor_list_url 
    }