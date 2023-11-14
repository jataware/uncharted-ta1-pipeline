from pathlib import Path
import os
import re
import json
from pydantic import BaseModel
from typing import Iterator, List, Tuple, Sequence, Union
from PIL.Image import Image as PILImage
from PIL import Image
import boto3


# regex for matching s3 uris
S3_URI_MATCHER = re.compile(r"^s3://[a-zA-Z0-9.-]+$")


class ImageFileInputIterator(Iterator[Tuple[str, PILImage]]):
    """Generates an iterable list of PIL images from a directory of images"""

    def __init__(self, image_path: Path) -> None:
        """Initializes the iterator"""
        self._image_files: List[Path] = []
        self._index = 0

        # recursivley traverse the input directory and find all image files
        for root, _, files in os.walk(image_path):
            for file in files:
                self._image_files.append(Path(os.path.join(root, file)))

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, PILImage]:
        """Loads the next image in the list of images"""
        if self._index < len(self._image_files):
            image_path = self._image_files[self._index]
            image = Image.open(image_path)
            doc_id = image_path.stem
            self._index += 1
            return (doc_id, image)
        else:
            raise StopIteration


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
