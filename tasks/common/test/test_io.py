import os
import os.path as path
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from typing import Dict, List
import boto3
from moto import mock_s3
from tasks.common.io import ImageFileInputIterator, JSONFileWriter


class TestData(BaseModel):
    name: str
    color: str


def test_image_file_input_iterator_filesystem():
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
    iterator = ImageFileInputIterator(str(test_dir))

    # Iterate over the images and verify the results
    expected_images = [(image1_path.stem, image1), (image2_path.stem, image2)]
    for expected_doc_id, expected_image in expected_images:
        doc_id, image = next(iterator)
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


# add a test for s3 input iterator
@mock_s3
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
    for expected_doc_id, expected_image in expected_images:
        doc_id, image = next(iterator)
        assert doc_id == expected_doc_id
        assert list(image.getdata()) == list(expected_image.getdata())

    # Verify that StopIteration is raised when there are no more images
    try:
        next(iterator)
        assert False, "StopIteration should have been raised"
    except StopIteration:
        pass


def test_json_file_writer_s3_uri():
    # Initialize the ImageFileInputIterator with the test directory
    image_file_input_iterator_s3("s3://test-bucket/data")


def test_json_file_writer_s3_url():
    # Initialize the ImageFileInputIterator with the test directory
    image_file_input_iterator_s3("https://buckets.com/test-bucket/data")


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
