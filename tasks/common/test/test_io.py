import os
import json
from pathlib import Path
from PIL import Image
from tasks.common.io import ImageFileInputIterator, JSONFileWriter
from pydantic import BaseModel
from typing import Dict, List


class TestData(BaseModel):
    name: str
    color: str


def test_image_file_input_iterator():
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


def test_json_file_writer():
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
