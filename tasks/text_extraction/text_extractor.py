import numpy as np
import os
import logging
import json
from math import ceil
from typing import Tuple, List, Dict, Any
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from .ocr.google_vision_ocr import GoogleVisionOCR
from .entities import (
    DocTextExtraction,
    TextExtraction,
    Point,
    Tile,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from ..common.task import Task, TaskInput, TaskResult

# ENV VARIABLE -- needed for google-vision API
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/path/to/google/vision/creds/json/file'

PIXEL_LIM_DEFAULT = 6000  # default max pixel limit for input image (determines amount of image resizing)

logger = logging.getLogger(__name__)


class TextExtractor(Task):
    """
    Base class for OCR-based text extraction
    """

    def __init__(
        self,
        task_id: str,
        cache_dir: Path,
        to_blocks: bool = True,
        document_ocr: bool = False,
    ):
        super().__init__(task_id, str(cache_dir))
        self._ocr = GoogleVisionOCR()
        self._model_id = "google-cloud-vision"
        self._to_blocks = to_blocks
        self._document_ocr = document_ocr

    def _extract_text(self, im: PILImage) -> List[Dict[str, Any]]:
        img_gv = GoogleVisionOCR.pil_to_vision_image(im)

        # ----- do GoogleVision OCR
        ocr_texts = []
        if self._document_ocr:
            ocr_texts = self._ocr.detect_document_text(img_gv)
        else:
            ocr_texts = self._ocr.detect_text(img_gv)
            if self._to_blocks:
                ocr_texts = self._ocr.text_to_blocks(ocr_texts)

        return ocr_texts

    def run(self, input: TaskInput) -> TaskResult:
        raise NotImplementedError


class ResizeTextExtractor(TextExtractor):
    """
    OCR-based text extraction with optional image scaling prior to OCR
    """

    def __init__(
        self,
        task_id: str,
        cache_dir: Path,
        to_blocks=True,
        document_ocr=False,
        pixel_lim: int = PIXEL_LIM_DEFAULT,
    ):
        super().__init__(task_id, cache_dir, to_blocks, document_ocr)
        self._pixel_lim = pixel_lim
        self._model_id += f"_resize-{pixel_lim}"

    def run(self, input: TaskInput) -> TaskResult:
        # im_orig_size = im.size   #(width, height)
        if input.image is None:
            return self._create_result(input)

        im_resized, im_resize_ratio = self._resize_image(input.image)

        doc_key = f"{input.raster_id}_{self._model_id}"

        # check cache and re-use existing file if present
        cached_json = self.fetch_cached_result(doc_key)
        if cached_json:
            result = self._create_result(input)
            result.add_output(
                TEXT_EXTRACTION_OUTPUT_KEY,
                DocTextExtraction(**cached_json).model_dump(),
            )
            return result

        ocr_blocks = self._extract_text(im_resized)

        # scale OCR pixel co-ords back to original image dimensions
        if ocr_blocks and im_resize_ratio < 1.0 and im_resize_ratio > 0.0:
            ocr_blocks = GoogleVisionOCR.scale_ocr_coords(
                ocr_blocks, 1.0 / im_resize_ratio
            )

        # convert output to internal schema
        texts: List[TextExtraction] = []
        for ocr_block in ocr_blocks:
            bounds = [
                Point(x=vertex.x, y=vertex.y)
                for vertex in ocr_block["bounding_poly"].vertices
            ]
            ocr_result = TextExtraction(
                text=ocr_block["text"], confidence=1.0, bounds=bounds
            )
            texts.append(ocr_result)

        doc_text_extraction = DocTextExtraction(doc_id=doc_key, extractions=texts)

        # write to cache
        self.write_result_to_cache(doc_text_extraction.model_dump(), doc_key)

        result = self._create_result(input)
        result.add_output(TEXT_EXTRACTION_OUTPUT_KEY, doc_text_extraction.model_dump())
        return result

    def _resize_image(self, im: PILImage) -> Tuple[PILImage, float]:
        """
        Resize an image, if needed, so max dimension is <= self._pixel_lim
        """
        # TODO could be moved to a 'common' module?

        im_orig_size = im.size  # (width, height)
        im_resize_ratio = 1.0
        if max(im_orig_size) > self._pixel_lim:
            im_resize_ratio = self._pixel_lim / max(im_orig_size)
            logger.info("Resizing image with ratio: {}".format(im_resize_ratio))

            reduced_size = int(im_orig_size[0] * im_resize_ratio), int(
                im_orig_size[1] * im_resize_ratio
            )
            im = im.resize(reduced_size, Image.Resampling.LANCZOS)

        return im, im_resize_ratio


class TileTextExtractor(TextExtractor):
    """
    OCR-based text extraction with image tiling prior to OCR
    """

    def __init__(
        self, task_id: str, cache_dir: Path, split_lim: int = PIXEL_LIM_DEFAULT
    ):
        super().__init__(task_id, cache_dir)
        self.split_lim = split_lim
        self._model_id += f"_tile-{split_lim}"

    def run(self, input: TaskInput) -> TaskResult:
        """
        Run OCR-based text extractor
        Image may be internally tiled prior to OCR, if needed

        Args:
            input: TaskInput object with image to process
        Returns:
            TaskResult object containing a DocTextExtraction object
        """

        # TODO -- this code could be modified to include overlap/stride len, etc.
        # (then, any overlapping OCR results need to be de-dup'd)
        if input.image is None:
            return self._create_result(input)

        doc_key = f"{input.raster_id}_{self._model_id}"

        # check cache and re-use existing file if present
        json_data = self.fetch_cached_result(doc_key)
        if json_data:
            result = self._create_result(input)
            result.add_output(
                TEXT_EXTRACTION_OUTPUT_KEY,
                DocTextExtraction(**json_data).model_dump(),
            )
            return result

        im_tiles = self._split_image(input.image, self.split_lim)
        logger.info(
            f"Image split into {len(im_tiles)} tiles. Extracting OCR text from each..."
        )

        ocr_blocks: List[Dict[str, Any]] = (
            []
        )  # list for OCR results across all tiles (whole image)
        for tile in im_tiles:
            # get OCR results for this tile
            tile_ocr_blocks = self._extract_text(tile.image)
            # convert OCR poly-bounds to global pixel coords and add to results
            ocr_blocks.extend(
                GoogleVisionOCR.offset_ocr_coords(tile_ocr_blocks, tile.coordinates)
            )

        # convert OCR results to TA1 schema
        texts: List[TextExtraction] = []
        for ocr_block in ocr_blocks:
            bounds = [
                Point(x=vertex.x, y=vertex.y)
                for vertex in ocr_block["bounding_poly"].vertices
            ]
            ocr_result = TextExtraction(
                text=ocr_block["text"], confidence=1.0, bounds=bounds
            )
            texts.append(ocr_result)

        doc_text_extraction = DocTextExtraction(doc_id=doc_key, extractions=texts)
        json_data = doc_text_extraction.model_dump()

        # write to cache
        self.write_result_to_cache(json_data, doc_key)

        result = self._create_result(input)
        result.add_output(TEXT_EXTRACTION_OUTPUT_KEY, json_data)
        return result

    def _split_image(self, image: PILImage, size_limit: int) -> List[Tile]:
        """
        split an image as needed to fit under the image size limit for x and y
        """

        image_size = image.size
        splits_x = self._get_splits(image_size[0], size_limit)
        splits_y = self._get_splits(image_size[1], size_limit)
        images: List[Tile] = []
        for split_y in splits_y:
            for split_x in splits_x:
                ims = Image.new(
                    mode="RGB", size=(split_x[1] - split_x[0], split_y[1] - split_y[0])
                )
                cropping = image.crop((split_x[0], split_y[0], split_x[1], split_y[1]))
                ims.paste(cropping, (0, 0))
                images.append(Tile(ims, (split_x[0], split_y[0])))
        return images

    def _get_splits(self, size: int, limit: int) -> List[Tuple]:
        """
        get the pixel intervals for image tiling
        note, currently the tile stride == limit (0% overlap)
        """
        splits = ceil(float(size) / limit)
        split_inc = ceil(float(size) / splits)
        split_vals: List[Tuple[int, float]] = []
        current = 0
        while current < size:
            next_inc = min(current + split_inc, size)
            split_vals.append((current, next_inc))
            current = next_inc
        return split_vals
