import logging
import os
from typing import Optional
from PIL.Image import Image as PILImage
from tasks.common.io import (
    ImageFileReader,
    ImageFileWriter,
    Mode,
    append_to_cache_location,
    bucket_exists,
    get_file_source,
)

logger = logging.getLogger(__name__)


class ImageCache:
    def __init__(self, cache_location: str) -> None:
        self._cache_location = cache_location
        self._image_reader = ImageFileReader()
        self._image_writer = ImageFileWriter()
        self._init_cache()

    def _init_cache(self):
        """
        If working with the file system, create local cache dir if it doesn't exist.  If working with S3,
        ensure the bucket exists and is accessible.
        """
        cache_mode = get_file_source(self._cache_location)
        if cache_mode == Mode.FILE:
            if not os.path.exists(self._cache_location):
                os.makedirs(self._cache_location)
        elif cache_mode == Mode.S3_URI or Mode.URL:
            if not bucket_exists(self._cache_location):
                raise Exception(
                    f"S3 cache bucket {self._cache_location} does not exist"
                )
        else:
            raise Exception(f"Invalid cache location {self._cache_location}")

    def _get_cache_doc_path(self, doc_key: str) -> str:
        """
        Generate the full local path for cached json result
        """
        return append_to_cache_location(self._cache_location, f"{doc_key}")

    def fetch_cached_result(self, doc_key: str) -> Optional[PILImage]:
        """
        Check if task result is available in the local cache
        """
        cached_path = self._get_cache_doc_path(doc_key)
        if not self._cache_location:
            return None
        try:
            return self._image_reader.process(cached_path)
        except Exception:
            logger.exception(
                f"error fetching cached image from {cached_path}", exc_info=True
            )
            return None

    def write_result_to_cache(self, image: PILImage, doc_key: str):
        """
        Write task result to local cache
        """
        if not self._cache_location:
            return
        doc_path = self._get_cache_doc_path(doc_key)
        self._image_writer.process(doc_path, image)
