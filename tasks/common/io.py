import io
import os
import re
import json
from urllib.parse import urlparse
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from typing import Iterator, List, Tuple, Sequence, Union
from PIL.Image import Image as PILImage
from PIL import Image
import boto3

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = 400000000  # to allow PIL to load large images

# regex for matching s3 uris
S3_URI_MATCHER = re.compile(r"^s3://[a-zA-Z0-9.-]+$")


class Mode(Enum):
    FILE = 1
    S3_URI = 2
    URL = 3


class ImageFileInputIterator(Iterator[Tuple[str, PILImage]]):
    """Generates an iterable list of PIL images from a directory of images"""

    def __init__(self, image_path: str) -> None:
        """Initializes the iterator"""

        self._image_files: List[str] = []
        self._index = 0

        # check if the string is an s3 uri or a file path and collect up the
        # locations of the files to be loaded
        mode = self._get_file_source(image_path)
        if mode == Mode.S3_URI or mode == Mode.URL:
            self._s3_init(image_path)
        else:
            self._file_init(image_path)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, PILImage]:
        """Loads the next image in the list of images"""
        if self._index < len(self._image_files):
            image_path = self._image_files[self._index]
            self._index += 1

            mode = self._get_file_source(image_path)
            if mode == Mode.S3_URI or mode == Mode.URL:
                # process the image from s3
                image = self._load_s3(image_path)
                doc_id = image_path.split("/")[-1].split(".")[0]
                return (doc_id, image)
            elif self._verify_is_image(Path(image_path)):
                # process the image from the local file system
                image = self._load_file(image_path)
                doc_id = image_path.split("/")[-1].split(".")[0]
                return (doc_id, image)
            return self.__next__()
        else:
            raise StopIteration

    def _load_file(self, path: str) -> PILImage:
        """Loads an image file into memory as a PIL image object"""
        return Image.open(path)

    def _load_s3(self, path: str) -> PILImage:
        """Loads an image file from s3 into memory as a PIL image object"""
        # extract bucket and prefix string from path
        bucket = path.split("/")[3]
        key = "/".join(path.split("/")[4:])
        # load image from s3
        s3 = boto3.resource("s3")
        obj = s3.Object(bucket, key)
        img_bytes_io = io.BytesIO(obj.get()["Body"].read())
        image = Image.open(img_bytes_io)
        return image

    def _get_file_source(self, path: str) -> Mode:
        """Checks if the path is a file, s3 uri, or url"""
        if S3_URI_MATCHER.match(path):
            return Mode.S3_URI
        elif urlparse(path).scheme == "http" or urlparse(path).scheme == "https":
            return Mode.URL
        else:
            return Mode.FILE

    def _file_init(self, path: str):
        """Initializes the iterator with a list of local image files"""

        # recursivley traverse the input directory and find all image files
        for root, _, files in os.walk(path):
            for file in files:
                self._image_files.append(os.path.join(root, file))
        self._image_files.sort()

    def _s3_init(self, path: str):
        """Initializes the iterator with a list of s3 image files"""

        # create s3 client
        client = boto3.client("s3")  # type: ignore
        # extract bucket and prefix string from path
        split_path = path.split("/")
        if self._get_file_source(path) == Mode.URL:
            parsed_url = urlparse(path)
            source = parsed_url.scheme + "://" + parsed_url.netloc
            bucket = split_path[3]
            prefix = "/".join(split_path[4:])
        else:
            source = "s3"
            bucket = split_path[2]
            prefix = "/".join(split_path[3:])

        # list objects in bucket with prefix
        objects = client.list_objects(Bucket=bucket, Prefix=prefix)
        # add all objects to the list of image files
        for obj in objects["Contents"]:
            if "Key" in obj:
                key = obj["Key"]
                self._image_files.append(f"{source}/{bucket}/{key}")
        self._image_files.sort()

    def _verify_is_image(self, image_path: Path) -> bool:
        """Verifies that the file at the given path is an image"""
        try:
            im = Image.open(image_path)
            im.verify()
            im.close()
            im = Image.open(image_path)
            im.transpose(Image.FLIP_LEFT_RIGHT)
            im.close()
            return True
        except:
            return False


class JSONFileWriter:
    """Writes a BaseModel as a JSON file to either the local file system or an s3 bucket"""

    def process(
        self, output_location: str, data: Union[BaseModel, Sequence[BaseModel]]
    ) -> None:
        """Writes metadata to a json file on the local file system or to an s3 bucket based
        on the output location format"""

        # check to see if path is an s3 uri - otherwise treat it as a file path
        if S3_URI_MATCHER.match(output_location):
            self._write_to_s3(data, output_location)
        else:
            self._write_to_file(data, Path(output_location))

    @staticmethod
    def _write_to_file(
        data: Union[BaseModel, Sequence[BaseModel]], output_location: Path
    ) -> None:
        """Writes metadata to a json file"""

        # get the director of the file
        if output_location.is_dir():
            raise ValueError(f"Output location {output_location} is not a file.")

        output_dir = output_location.parent
        if not output_dir.exists():
            os.makedirs(output_dir)

        # write the data to the output file
        with open(output_location, "w") as outfile:
            if isinstance(data, Sequence):
                for d in data:
                    json.dump(d.model_dump(), outfile)
                    outfile.write("\n")
            else:
                json.dump(data.model_dump(), outfile)

    @staticmethod
    def _write_to_s3(
        data: Union[BaseModel, Sequence[BaseModel]], output_uri: str
    ) -> None:
        """Writes metadata to an s3 bucket"""

        # create s3 client
        client = boto3.client("s3")

        # extract bucket from s3 uri
        bucket = output_uri.split("/")[2]

        # write data to the bucket
        if isinstance(data, Sequence):
            model_list = [d.model_dump() for d in data]
            model_str = json.dumps(model_list)
            client.put_object(
                Body=bytes(model_str.encode("utf-8")), Bucket=bucket, Key=output_uri
            )
        else:
            json_model = data.model_dump_json()
            client.put_object(
                Body=bytes(json_model.encode("utf-8")),
                Bucket=bucket,
                Key=output_uri,
            )
