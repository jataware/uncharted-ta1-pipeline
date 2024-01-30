import os
import os.path as path
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from cv2 import exp
from pydantic import BaseModel
from typing import Dict, List
import boto3
from moto.core.dectorator import mock_aws
from tasks.common.io import ImageFileInputIterator, JSONFileWriter, ImageFileWriter


class TestData(BaseModel):
    name: str
    color: str


def image_file_input_iterator_filesystem(test_path: str):
    # Create a temporary directory and save some test images
    test_dir = Path("tasks/common/test/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    image1_path = test_dir / "image1.png"
    image2_path = test_dir / "image2.png"
    image1 = Image.new("RGB", (100, 100), color="red")
    image2 = Image.new("RGB", (100, 100), color="blue")
    image1.save(image1_path)
    image2.save(image2_path)

    # Initialize the ImageFileInputIterator with the test directory
    iterator = ImageFileInputIterator(str(test_path))

    # Iterate over the images and verify the results
    expected_images = [(image1_path.stem, image1), (image2_path.stem, image2)]
    for doc_id, image in iterator:
        expected_doc_id, expected_image = expected_images.pop(0)
        assert doc_id == expected_doc_id
        assert list(image.getdata()) == list(expected_image.getdata())

    # Verify that StopIteration is raised when there are no more images
    try:
        next(iterator)
        assert False, "StopIteration should have been raised"
    except StopIteration:
        pass

    # Clean up the temporary directory
    image1_path.unlink()
    image2_path.unlink()
    test_dir.rmdir()


def test_image_file_input_iterator_filesystem_file():
    image_file_input_iterator_filesystem("tasks/common/test/data/image1.png")


def test_image_file_input_iterator_filesystem_dir():
    image_file_input_iterator_filesystem("tasks/common/test/data")


# add a test for s3 input iterator
@mock_aws
def image_file_input_iterator_s3(input_path: str):
    # Create a temporary directory and save some test images
    test_prefix = "data"
    test_bucket = "test-bucket"

    conn = boto3.resource("s3", region_name="us-east-1")
    bucket = conn.create_bucket(Bucket=test_bucket)

    image1 = Image.new("RGB", (100, 100), color="red")
    image1_io = BytesIO()
    image1.save(image1_io, format="png")
    image1_io.seek(0)

    image2 = Image.new("RGB", (100, 100), color="blue")
    image2_io = BytesIO()
    image2.save(image2_io, format="png")
    image2_io.seek(0)

    bucket.put_object(
        Key=path.join(test_prefix, "image1.png"),
        Body=image1_io,
    )
    bucket.put_object(
        Key=path.join(test_prefix, "image2.png"),
        Body=image2_io,
    )

    # Iterate over the images and verify the results
    iterator = ImageFileInputIterator(input_path)
    expected_images = [
        ("image1", image1),
        ("image2", image2),
    ]
    for doc_id, image in iterator:
        expected_doc_id, expected_image = expected_images.pop(0)
        assert doc_id == expected_doc_id
        assert list(image.getdata()) == list(expected_image.getdata())

    # Verify that StopIteration is raised when there are no more images
    try:
        next(iterator)
        assert False, "StopIteration should have been raised"
    except StopIteration:
        pass


def test_image_file_iterator_s3_uri():
    # Initialize the ImageFileInputIterator with the test directory
    image_file_input_iterator_s3("s3://test-bucket/data")
    image_file_input_iterator_s3("s3://test-bucket/data/image1.png")


def test_image_file_iterator_s3_url():
    # Initialize the ImageFileInputIterator with the test directory
    image_file_input_iterator_s3("https://buckets.com/test-bucket/data")
    image_file_input_iterator_s3("https://buckets.com/test-bucket/data/image1.png")


def test_json_file_writer_filesystem():
    # Test writing a single BaseModel instance
    test_data = TestData(name="test", color="red")
    output_location = "tasks/common/test/data/test.json"
    writer = JSONFileWriter()
    writer.process(output_location, test_data)

    with open(output_location, "r") as f:
        data = json.load(f)
        assert data == {"name": "test", "color": "red"}

    os.remove(output_location)

    # Test writing a list of BaseModel instances
    test_data = [
        TestData(name="test1", color="red"),
        TestData(name="test2", color="blue"),
    ]
    output_location = "tasks/common/test/data/test.json"
    writer.process(output_location, test_data)

    data: List[Dict[str, str]] = []
    with open(output_location, "r") as f:
        for line in f:
            data.append(json.loads(line))
        assert data == [
            {"name": "test1", "color": "red"},
            {"name": "test2", "color": "blue"},
        ]

    os.remove(output_location)


@mock_aws
def test_json_file_writer_s3():
    # Create a temporary directory and save some test images
    test_bucket = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    bucket = conn.create_bucket(Bucket=test_bucket)

    # Test writing a single BaseModel instance
    test_data = TestData(name="test", color="red")
    output_location = "s3://test-bucket/data/test.json"
    writer = JSONFileWriter()
    writer.process(output_location, test_data)

    obj = bucket.Object("data/test.json")
    data = json.loads(obj.get()["Body"].read())
    assert data == {"name": "test", "color": "red"}

    # Test writing a list of BaseModel instances
    test_data = [
        TestData(name="test1", color="red"),
        TestData(name="test2", color="blue"),
    ]
    writer.process(output_location, test_data)

    obj = bucket.Object("data/test.json")
    data = []
    for line in obj.get()["Body"].read().decode("utf-8").splitlines():
        data.append(json.loads(line))
    assert data == [
        {"name": "test1", "color": "red"},
        {"name": "test2", "color": "blue"},
    ]


@mock_aws
def test_image_writer_s3():
    # Create a temporary directory and save a test image
    test_bucket = "test-bucket"
    conn = boto3.resource("s3", region_name="us-east-1")
    bucket = conn.create_bucket(Bucket=test_bucket)

    test_data = Image.new("RGB", (100, 100), color="red")
    output_location = "s3://test-bucket/data/test.png"
    writer = ImageFileWriter()
    writer.process(output_location, test_data)

    # read the image back from s3 and verify it is the same
    obj = bucket.Object("data/test.png")
    bytes_io = BytesIO(obj.get()["Body"].read())
    image = Image.open(bytes_io)

    assert image.tobytes() == test_data.tobytes()


def test_image_writer_filesystem():
    # Create a temporary directory and save a test image
    test_dir = Path("tasks/common/test/data")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_data = Image.new("RGB", (100, 100), color="red")
    output_location = "tasks/common/test/data/test.png"
    writer = ImageFileWriter()
    writer.process(output_location, test_data)

    # read the image back from s3 and verify it is the same
    image = Image.open(output_location)
    assert image.tobytes() == test_data.tobytes()

    # clean up the temporary directory
    os.remove(output_location)
    test_dir.rmdir()
