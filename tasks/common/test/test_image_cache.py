import os
from PIL.Image import Image as PILImage
from PIL import Image
import boto3
from moto import mock_aws
import pytest
from tasks.common.image_cache import ImageCache


@pytest.fixture
def local_cache_dir(tmp_path):
    return tmp_path / "image_cache"


@pytest.fixture
def s3_bucket_name():
    return "test-bucket"


@pytest.fixture
def s3_cache_location(s3_bucket_name):
    return f"s3://{s3_bucket_name}/image_cache"


@pytest.fixture
def image():
    return Image.new("RGB", (100, 100), color="red")


def test_init_cache_local(local_cache_dir):
    cache = ImageCache(str(local_cache_dir))
    assert os.path.exists(local_cache_dir)


@mock_aws
def test_init_cache_s3(s3_cache_location: str, s3_bucket_name: str) -> None:
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=s3_bucket_name)
    cache = ImageCache(s3_cache_location)
    assert cache._cache_location == s3_cache_location


def test_fetch_cached_result_local(local_cache_dir, image):
    cache = ImageCache(str(local_cache_dir))
    doc_key = "test_image"
    cache.write_result_to_cache(image, doc_key)
    fetched_image = cache.fetch_cached_result(doc_key)
    assert fetched_image
    assert fetched_image.tobytes() == image.tobytes()


@mock_aws
def test_fetch_cached_result_s3(
    s3_cache_location: str, s3_bucket_name: str, image: PILImage
) -> None:
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=s3_bucket_name)
    cache = ImageCache(s3_cache_location)
    doc_key = "test_image"
    cache.write_result_to_cache(image, doc_key)
    fetched_image = cache.fetch_cached_result(doc_key)
    assert fetched_image
    assert fetched_image.tobytes() == image.tobytes()


def test_write_result_to_cache_local(local_cache_dir, image):
    cache = ImageCache(str(local_cache_dir))
    doc_key = "test_image"
    cache.write_result_to_cache(image, doc_key)
    doc_path = cache._get_cache_doc_path(doc_key)
    assert os.path.exists(doc_path)


@mock_aws
def test_write_result_to_cache_s3(s3_cache_location, s3_bucket_name, image):
    conn = boto3.resource("s3", region_name="us-east-1")
    conn.create_bucket(Bucket=s3_bucket_name)
    cache = ImageCache(s3_cache_location)
    doc_key = "test_image.json"
    cache.write_result_to_cache(image, doc_key)
    doc_path = cache._get_cache_doc_path(doc_key)
    bucket = conn.Bucket(s3_bucket_name)
    o = list(bucket.objects.filter())
    objs = list(bucket.objects.filter(Prefix="image_cache/test_image.json"))
    assert len(objs) == 1
