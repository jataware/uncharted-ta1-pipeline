import io
import logging
import os
import re
import json
from urllib.parse import urlparse
from enum import Enum
from pathlib import Path
import httpx
from pydantic import BaseModel
from typing import Any, Dict, Iterator, List, Tuple
from PIL.Image import Image as PILImage
from PIL import Image
from tasks.common.image_io import normalize_image_format
import boto3

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = 400000000  # to allow PIL to load large images

# regex for matching s3 uris
S3_URI_MATCHER = re.compile(r"^s3://[a-zA-Z0-9.\-_]{1,255}/.*$")

logger = logging.getLogger(__name__)


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
        mode = get_file_source(image_path)
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

            mode = get_file_source(image_path)
            if mode == Mode.S3_URI or mode == Mode.URL:
                # process the image from s3
                image = self._load_s3(image_path, mode)
                image = normalize_image_format(image)
                doc_id = image_path.split("/")[-1].split(".")[0]
                return (doc_id, image)
            elif self._verify_is_image(Path(image_path)):
                # process the image from the local file system
                image = self._load_file(image_path)
                image = normalize_image_format(image)
                doc_id = image_path.split("/")[-1].split(".")[0]
                return (doc_id, image)
            return self.__next__()
        else:
            raise StopIteration

    def _load_file(self, path: str) -> PILImage:
        """Loads an image file into memory as a PIL image object"""
        return Image.open(path)

    def _load_s3(self, path: str, mode: Mode) -> PILImage:
        """Loads an image file from s3 into memory as a PIL image object"""
        # extract bucket and prefix string from path
        bucket, key = parse_s3_reference(path, mode)
        # load image from s3
        s3 = boto3.resource("s3")
        obj = s3.Object(bucket, key)
        img_bytes_io = io.BytesIO(obj.get()["Body"].read())
        image = Image.open(img_bytes_io)
        return image

    def _file_init(self, path: str):
        """Initializes the iterator with a list of local image files"""

        # recursivley traverse the input directory and find all image files, or
        # add the single file to the list of image files
        path_obj = Path(path)
        if path_obj.is_dir():
            for root, _, files in os.walk(path):
                for file in files:
                    self._image_files.append(os.path.join(root, file))
        else:
            self._image_files.append(path)
        self._image_files.sort()

    def _s3_init(self, path: str):
        """Initializes the iterator with a list of s3 image files"""

        # create s3 client
        client = boto3.client("s3")  # type: ignore
        # extract bucket and prefix string from path
        split_path = path.split("/")
        if get_file_source(path) == Mode.URL:
            parsed_url = urlparse(path)
            source = parsed_url.scheme + "://" + parsed_url.netloc
            bucket = split_path[3]
            prefix = "/".join(split_path[4:])
        else:
            source = "s3:/"
            bucket = split_path[2]
            prefix = "/".join(split_path[3:])

        # list objects in bucket with prefix
        objects = client.list_objects(Bucket=bucket, Prefix=prefix)
        # add all objects to the list of image files
        if "Contents" not in objects:
            raise Exception("missing s3 content")
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
            im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            im.close()
            return True
        except:
            return False


class JSONFileWriter:
    """Writes a BaseModel as a JSON file to either the local file system or an s3 bucket"""

    def process(self, output_location: str, data: BaseModel) -> None:
        """Writes metadata to a json file on the local file system or to an s3 bucket based
        on the output location format"""

        # check to see if path is an s3 uri - otherwise treat it as a file path
        source = get_file_source(output_location)
        if source == Mode.S3_URI or source == Mode.URL:
            self._write_to_s3(data, output_location)
        else:
            self._write_to_file(data, Path(output_location))

    @staticmethod
    def _write_to_file(data: BaseModel, output_location: Path) -> None:
        """Writes metadata to a json file"""

        # get the director of the file
        if output_location.is_dir():
            raise ValueError(f"Output location {output_location} is not a file.")

        output_dir = output_location.parent
        if not output_dir.exists():
            os.makedirs(output_dir)

        # write the data to the output file
        with open(output_location, "w") as outfile:
            json.dump(data.model_dump(), outfile)

    @staticmethod
    def _write_to_s3(data: BaseModel, output_uri: str) -> None:
        """Writes metadata to an s3 bucket"""

        # create s3 client based on the mode
        mode = get_file_source(output_uri)
        if mode == Mode.S3_URI:
            client = boto3.client("s3")
        elif mode == Mode.URL:
            parsed_url = urlparse(output_uri)
            client = boto3.client(
                "s3", endpoint_url=f"{parsed_url.scheme}://{parsed_url.netloc}"
            )

        # extract bucket from s3 uri
        bucket, key = parse_s3_reference(output_uri, mode)

        # write data to the bucket
        json_model = data.model_dump_json()
        client.put_object(
            Body=bytes(json_model.encode("utf-8")),
            Bucket=bucket,
            Key=key,
        )


class BytesIOFileWriter:
    """Write bytes to a file on the local file system or an s3 bucket"""

    def process(self, output_location: str, data: io.BytesIO) -> None:
        """Writes bytes to a file on the local file system or to an s3 bucket based
        on the output location format"""

        # check to see if path is an s3 uri - otherwise treat it as a file path
        if S3_URI_MATCHER.match(output_location):
            self._write_to_s3(data, output_location)
        else:
            self._write_to_file(data, Path(output_location))

    @staticmethod
    def _write_to_file(data: io.BytesIO, output_location: Path) -> None:
        """Writes bytes to a file"""

        # get the director of the file
        if output_location.is_dir():
            raise ValueError(f"Output location {output_location} is not a file.")

        output_dir = output_location.parent
        if not output_dir.exists():
            os.makedirs(output_dir)

        # write the data to the output file
        with open(output_location, "wb") as outfile:
            outfile.write(data.getvalue())

    @staticmethod
    def _write_to_s3(data: io.BytesIO, output_uri: str) -> None:
        """Writes bytes to an s3 bucket"""

        # create s3 client
        client = boto3.client("s3")  # type: ignore

        # extract bucket from s3 uri
        bucket = output_uri.split("/")[2]
        key = "/".join(output_uri.split("/")[3:])

        # write data to the bucket
        client.put_object(Body=data.getvalue(), Bucket=bucket, Key=key)


class ImageFileWriter(BytesIOFileWriter):
    """Writes a PIL image to either the local file system or an s3 bucket"""

    def process(self, output_location: str, image: PILImage) -> None:
        """Writes an image to a file on the local file system or to an s3 bucket based
        on the output location format"""

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        # check to see if path is an s3 uri - otherwise treat it as a file path
        if S3_URI_MATCHER.match(output_location):
            self._write_to_s3(buf, output_location)
        else:
            self._write_to_file(buf, Path(output_location))


def download_file(image_url: str) -> bytes:
    r = httpx.get(image_url, timeout=1000)
    return r.content


class JSONFileReader:
    """Reads a JSON file and returns a list of BaseModel objects"""

    def process(self, input_location: str) -> Dict[Any, Any]:
        """Reads a JSON file and returns a list of BaseModel objects"""

        # check to see if path is an s3 uri - otherwise treat it as a file path
        source = get_file_source(input_location)
        if source == Mode.S3_URI or source == Mode.URL:
            return self._read_from_s3(input_location, source)
        else:
            return self._read_from_file(Path(input_location))

    @staticmethod
    def _read_from_file(input_location: Path) -> Dict[Any, Any]:
        """Reads a JSON file and returns a list of BaseModel objects"""

        # get the director of the file
        if input_location.is_dir():
            raise ValueError(f"Input location {input_location} is not a file.")

        # read the data from the input file
        with open(input_location, "r") as infile:
            data = json.load(infile)
        return data

    @staticmethod
    def _read_from_s3(input_uri: str, mode: Mode) -> Dict[Any, Any]:
        """Reads a JSON file from an s3 bucket and returns a list of BaseModel objects"""

        # create an s3 client based on the mode
        if mode == Mode.S3_URI:
            client = boto3.client("s3")
        elif mode == Mode.URL:
            parsed_url = urlparse(input_uri)
            client = boto3.client(
                "s3", endpoint_url=f"{parsed_url.scheme}://{parsed_url.netloc}"
            )
        else:
            raise ValueError(
                f"Invalid URI mode for S3 client instantiation: {input_uri}"
            )

        # extract bucket from s3 uri
        bucket, key = parse_s3_reference(input_uri, mode)

        # read data from the bucket
        response = client.get_object(Bucket=bucket, Key=key)
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise Exception(f"Failed to read from s3 bucket {bucket} with key {key}.")

        data = json.loads(response["Body"].read().decode("utf-8"))
        return data


def parse_s3_reference(path: str, mode: Mode) -> Tuple[str, str]:
    """Parses the S3 reference to extract the bucket and key."""
    if mode == Mode.S3_URI:
        # For S3 URI, e.g., s3://lara/cache/text
        parsed_url = urlparse(path)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip("/")
    elif mode == Mode.URL:
        # For URL, e.g., https://s3.t1.uncharted/lara/cache/text
        parsed_url = urlparse(path)
        path_parts = parsed_url.path.split("/")
        bucket = path_parts[1]
        key = "/".join(path_parts[2:])
    else:
        raise ValueError(f"Invalid mode {mode}")
    return (bucket, key)


def get_file_source(path: str) -> Mode:
    """Checks if the path is a file, s3 uri, or url"""
    if S3_URI_MATCHER.match(path):
        return Mode.S3_URI
    elif urlparse(path).scheme == "http" or urlparse(path).scheme == "https":
        return Mode.URL
    else:
        return Mode.FILE


def bucket_exists(uri: str) -> bool:
    """Check if the bucket exists and is accessible"""
    mode = get_file_source(uri)
    if mode == Mode.S3_URI:
        client = boto3.client("s3")
    elif mode == Mode.URL:
        parsed_url = urlparse(uri)
        client = boto3.client(
            "s3", endpoint_url=f"{parsed_url.scheme}://{parsed_url.netloc}"
        )
    else:
        raise ValueError(f"Invalid URI mode for S3 client instantiation: {uri}")

    bucket, _ = parse_s3_reference(uri, mode)
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except client.exceptions.NoSuchBucket:
        return False
    except client.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404 or error_code == 403:
            return False
        raise


def append_to_cache_location(cache_location: str, append_str: str) -> str:
    """Appends a string to the cache location"""
    if cache_location[-1] == "/":
        return cache_location + append_str
    return cache_location + "/" + append_str
