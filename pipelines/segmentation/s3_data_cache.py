
import logging, os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

logger = logging.getLogger(__name__)


class S3DataCache():
    '''
    Class to handle downloading data from S3
    and caching to local disk
    '''

    def __init__(self, 
                 local_cache_path: str,
                 s3_url: str,
                 s3_bucket: str='lara',
                 aws_region_name: str=None,
                 aws_access_key_id: str=None,
                 aws_secret_access_key=None):
        
        self.s3_bucket = s3_bucket

        if not local_cache_path:
            raise Exception('local_cache_path not given for S3ModelCache')
        
        # create local cache dirs, if needed
        self.local_cache_path = local_cache_path
        if not os.path.exists(self.local_cache_path):
            os.makedirs(self.local_cache_path)  

        if s3_url:
            logger.info(f'Connecting to S3 at {s3_url}')
        
            self.s3_resource = boto3.resource(service_name='s3',
                        endpoint_url=s3_url,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)

            if self.s3_bucket:
                if not S3DataCache.bucket_exists(self.s3_resource, self.s3_bucket):
                    raise Exception('S3 bucket {} does not exist'.format(s3_bucket))
            else:
                logger.warning('S3 bucket name not given! Access to S3 not possible')
    

    def fetch_file_from_s3(self, s3_obj_key: str, overwrite: bool=False) -> str:
        '''
        Fetch file, S3 -> local disk
        (Overwrite is False by default)
        Returns local path where file is stored     
        '''

        if not self.s3_bucket:
            logger.warning(f'Skipping fetch_data_from_s3 {s3_obj_key}. No s3_bucket given')
            return ''

        # local path to store file
        filepath_local = os.path.join(self.local_cache_path.lstrip('/'), s3_obj_key)
        if Path(filepath_local).exists() and not overwrite:
            logger.info(f'Data already exists on local path {filepath_local}. Skipping data download from S3.')
            return filepath_local
        
        # create local dirs, if needed
        os.makedirs(filepath_local[:filepath_local.rfind('/')] , exist_ok=True)
        
        logger.info(f'Downloading data from s3 to local disk: {s3_obj_key}')
        S3DataCache.get_s3_object_to_file(self.s3_resource.Bucket(self.s3_bucket), s3_obj_key, filepath_local)

        return filepath_local
    

    def list_bucket_contents(self, path_prefix: str):
        '''
        List an S3 bucket contents based on key prefix
        Returns list of matching keys
        '''
        resp = self.s3_resource.meta.client.list_objects_v2(Bucket=self.s3_bucket, Prefix=path_prefix, MaxKeys=1000)
        if 'Contents' in resp:
            return [x['Key'] for x in resp['Contents']]
        else:
            return []


    @staticmethod
    def bucket_exists(s3_resource: boto3.resource, bucket_name: str) -> bool:
        '''
        Check if an s3 bucket exists
        '''
        exists = False
        try:
            s3_resource.meta.client.head_bucket(Bucket=bucket_name)
            print("Bucket {} exists.".format(bucket_name))
            exists = True
        except ClientError as e:
            print("Bucket {} doesn't exist or you don't have access to it.".format(bucket_name))
        return exists


    @staticmethod
    def get_s3_object_to_file(bucket, object_key: str, filename: str):
        '''
        Download an object from an s3 bucket and save to local file
        '''
        try:
            bucket.download_file(object_key, filename)
            print("Got object {} from bucket {} and saved to {}".format(object_key, bucket.name, filename))
        except ClientError as e:
            print("Error! Couldn't get and save object {} from bucket {}".format(object_key, bucket.name))
            print(repr(e))
            raise