from pathlib import Path
from moto import mock_aws
import boto3
from tasks.common.s3_data_cache import S3DataCache


def setup_s3_and_cache():
    # create an s3 data cache instance
    s3_data_cache = S3DataCache(
        s3_bucket="test-bucket",
        aws_region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        local_cache_path="/tmp",
    )

    # create a test bucket and file
    s3 = boto3.resource("s3", region_name="us-east-1")
    bucket = s3.create_bucket(Bucket="test-bucket")
    bucket.put_object(Key="test-file-1", Body=b"test body 1")
    bucket.put_object(Key="test-file-2", Body=b"test body 2")

    return s3_data_cache, ("test-file-1", "test-file-2")


@mock_aws
def test_fetch_file_from_s3():
    s3_data_cache, test_files = setup_s3_and_cache()

    # fetch the file from s3
    filepath_local = s3_data_cache.fetch_file_from_s3(test_files[0])

    # verify the file was fetched
    assert Path(filepath_local).exists()
    assert Path(filepath_local).read_text() == "test body 1"

    # clean up
    Path(filepath_local).unlink()


@mock_aws
def test_fetch_file_from_local_cache():
    s3_data_cache, test_files = setup_s3_and_cache()

    # fetch the file from s3
    filepath_local = s3_data_cache.fetch_file_from_s3(test_files[0])

    # verify the file was fetched
    assert Path(filepath_local).exists()
    assert Path(filepath_local).read_text() == "test body 1"

    # fetch the file again, this time it should come from the local cache
    filepath_local_cache = s3_data_cache.fetch_file_from_s3(test_files[0])

    # verify the file was fetched from the local cache
    assert filepath_local_cache == filepath_local

    # clean up
    Path(filepath_local).unlink()


@mock_aws
def test_list_bucket_contents():
    s3_data_cache, _ = setup_s3_and_cache()

    # create a test bucket and files
    s3 = boto3.resource("s3", region_name="us-east-1")
    bucket = s3.create_bucket(Bucket="test-bucket")
    bucket.put_object(Key="test-file-1", Body=b"test body 1")
    bucket.put_object(Key="test-file-2", Body=b"test body 2")

    # list the bucket contents
    contents = s3_data_cache.list_bucket_contents("test-file")

    # verify the contents
    assert set(contents) == {"test-file-1", "test-file-2"}
