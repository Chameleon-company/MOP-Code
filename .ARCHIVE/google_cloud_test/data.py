import io


class DataStorageFactory:
    def create(self, env):
        if env == 'aws':
            return AWSS3Repo()
        elif env == 'google':
            return GoogleCloudRepo()
        raise NameError('Missing env implementation')


class StorageRepo:
    def get(self, file):
        ''' Will fetch a file stream for an object stored in cloud storage '''
        pass


class AWSS3Repo(StorageRepo):
    def get(self, file):
        import boto3
        bucket = 'opendataplayground.deakin'
        s3 = boto3.resource('s3', region_name='ap-southeast-2')
        bucket = s3.Bucket(bucket)
        object = bucket.Object(file)
        response = object.get()
        file_stream = response['Body']
        return file_stream


class GoogleCloudRepo(StorageRepo):
    def get(self, file):
        from google.cloud import storage

        """Downloads a blob from the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"

        # The ID of your GCS object
        # source_blob_name = "storage-object-name"

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"

        storage_client = storage.Client()

        bucket = storage_client.bucket('melbourne_opendata_playground')
        # bucket.create(client=storage_client, project='D2IMELB',
        #               location='australia-southeast1')

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.

        # for blob in storage_client.list_blobs(bucket):
        #     print(blob.name)

        blob = bucket.blob(file)

        from tempfile import NamedTemporaryFile
        temp_download_file = NamedTemporaryFile().name

        blob.download_to_filename(temp_download_file)

        filestream = io.FileIO(temp_download_file)
        return filestream


if __name__ == "__main__":
    repo = DataStorageFactory().create('google')
    file = repo.get('parkingsensor/parkingsensor.csv')
    read_file = file.read()
    for byte in read_file:
        print(byte)
