import sys
import numpy as np
import logging
from math import ceil
from typing import Tuple, List, Dict, Any
from PIL import Image
from PIL.Image import Image as PILImage
import cv2
import requests
from .ocr.google_vision_ocr import GoogleVisionOCR
from .entities import (
    DocTextExtraction,
    TextExtraction,
    Point,
    Tile,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from ..common.task import Task, TaskInput, TaskResult

PIXEL_LIM_DEFAULT = 6000  # default max pixel limit for input image (determines amount of image resizing)

# default image gamma correction; =1 no change (disabled); <1 lightens the image; >1 darkens the image
# NOTE: gamma = 0.5 recommended for OCR pre-processing
GAMMA_CORR_DEFAULT = 1.0

logger = logging.getLogger(__name__)


class TextExtractor(Task):
    """
    Base class for OCR-based text extraction
    """

    def __init__(
        self,
        task_id: str,
        cache_location: str,
        to_blocks: bool = True,
        document_ocr: bool = False,
        gamma_correction: float = GAMMA_CORR_DEFAULT,
        output_key: str = TEXT_EXTRACTION_OUTPUT_KEY,
        metrics_url: str = "",
        cloud_authenticate=False,
    ):
        super().__init__(task_id, cache_location)
        self._ocr = GoogleVisionOCR(cloud_authenticate=cloud_authenticate)
        self._model_id = "google-cloud-vision"
        self._to_blocks = to_blocks
        self._document_ocr = document_ocr
        self._gamma_correction = gamma_correction
        self._output_key = output_key
        self._metrics_url = metrics_url

        # validate the vision api key - hard stop if can't be found
        try:
            self._ocr.validate_api_key()
        except Exception as e:
            logger.error(
                f"Google Vision OCR api validation failed with error: {repr(e)}"
            )
            sys.exit(1)

        # init gamma correction look up table
        self._gamma_lut = np.empty((1, 256), np.uint8)
        if self._gamma_correction != 1.0:
            # from https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
            for i in range(256):
                self._gamma_lut[0, i] = np.clip(
                    pow(i / 255.0, self._gamma_correction) * 255.0, 0, 255
                )

    def _apply_gamma_correction(self, img: PILImage) -> PILImage:
        """
        Apply image gamma correction prior to OCR
        """
        logger.info(f"applying gamma correction of {self._gamma_correction} to image")
        if self._gamma_correction == 1.0:
            # skip gamma correction
            return img

        # convert image from PIL to opencv (numpy) format --  assumed color channel order is RGB
        im = np.array(img)

        im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
        (L, A, B) = cv2.split(im)
        # apply gamma correction to L channel
        L_gamma = cv2.LUT(L, self._gamma_lut)
        # re-merge channels
        im_gamma = cv2.merge([L_gamma, A, B])
        im_gamma = cv2.cvtColor(im_gamma, cv2.COLOR_LAB2RGB)

        return Image.fromarray(im_gamma)

    def _extract_text(self, im: PILImage) -> List[Dict[str, Any]]:
        """
        Extract text from an image using Google Vision OCR.
        Args:
            im (PILImage): The input image from which text needs to be extracted.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the extracted text and related information.
        """

        img_gv = GoogleVisionOCR.pil_to_vision_image(im)

        # ----- do GoogleVision OCR
        ocr_texts = []
        if self._document_ocr:
            ocr_texts = self._ocr.detect_document_text(img_gv)
            self._publish_image_ocr_metrics()
        else:
            ocr_texts = self._ocr.detect_text(img_gv)
            if self._to_blocks:
                ocr_texts = self._ocr.text_to_blocks(ocr_texts)
                self._publish_doc_ocr_metrics()
        return ocr_texts

    def _publish_image_ocr_metrics(self, num_calls=1):
        """
        Publishes Vision image OCR  metrics to the specified metrics URL.
        Args:
            num_calls (int, optional): The number of OCR calls to report. Defaults to 1.
        Sends:
            - A POST request to update the total number of OCR calls.
            - A POST request to update the current number of OCR calls.
        """
        if self._metrics_url != "":
            requests.post(
                f"{self._metrics_url}/counter/total_image_ocr_calls?step={num_calls}"
            )
            requests.post(
                f"{self._metrics_url}/gauge/image_ocr_calls?value={num_calls}"
            )

    def _publish_doc_ocr_metrics(self, num_calls=1):
        """
        Publishes Vision document OCR metrics to the specified metrics URL.
        Args:
            num_calls (int, optional): The number of OCR calls to report. Defaults to 1.
        Sends:
            - A POST request to update the total number of OCR calls.
            - A POST request to update the current number of OCR calls.
        """
        if self._metrics_url != "":
            requests.post(
                f"{self._metrics_url}/counter/total_doc_ocr_calls?step={num_calls}"
            )
            requests.post(f"{self._metrics_url}/gauge/doc_ocr_calls?value={num_calls}")

    def run(self, input: TaskInput) -> TaskResult:
        raise NotImplementedError


class ResizeTextExtractor(TextExtractor):
    """
    OCR-based text extraction with optional image scaling prior to OCR
    """

    def __init__(
        self,
        task_id: str,
        cache_location: str,
        to_blocks=True,
        document_ocr=False,
        pixel_lim: int = PIXEL_LIM_DEFAULT,
        gamma_correction: float = GAMMA_CORR_DEFAULT,
        output_key: str = TEXT_EXTRACTION_OUTPUT_KEY,
        metrics_url: str = "",
    ):
        super().__init__(
            task_id,
            cache_location,
            to_blocks,
            document_ocr,
            gamma_correction,
            output_key,
            metrics_url,
        )
        self._pixel_lim = pixel_lim
        self._model_id += f"_resize-{pixel_lim}"

    def run(self, input: TaskInput) -> TaskResult:
        # im_orig_size = im.size   #(width, height)
        if input.image is None:
            return self._create_result(input)

        doc_key = f"{input.raster_id}_{self._model_id}_{self._gamma_correction}"

        # check cache and re-use existing file if present
        cached_data = self.fetch_cached_result(doc_key)
        if cached_data:
            # validate the cached data
            doc_extract = DocTextExtraction.model_validate(cached_data)
            result = self._create_result(input)
            result.add_output(self._output_key, doc_extract.model_dump())
            return result

        # pre-processing: apply gamma correction and re-size image, as needed
        im_resized, im_resize_ratio = self._resize_image(
            self._apply_gamma_correction(input.image)
        )

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
        self.write_result_to_cache(doc_text_extraction, doc_key)

        result = self._create_result(input)
        result.add_output(self._output_key, doc_text_extraction.model_dump())
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
        self,
        task_id: str,
        cache_location: str,
        split_lim: int = PIXEL_LIM_DEFAULT,
        gamma_correction: float = GAMMA_CORR_DEFAULT,
        output_key: str = TEXT_EXTRACTION_OUTPUT_KEY,
        metrics_url: str = "",
    ):
        super().__init__(
            task_id,
            cache_location,
            gamma_correction=gamma_correction,
            output_key=output_key,
            metrics_url=metrics_url,
        )
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

        doc_key = f"{input.raster_id}_{self._model_id}_{self._gamma_correction}"

        # check cache and re-use existing file if present
        cached_data = self.fetch_cached_result(doc_key)
        if cached_data:
            logger.info(f"Using cached OCR results for raster: {input.raster_id}")
            doc_text = DocTextExtraction.model_validate(cached_data)
            result = self._create_result(input)
            result.add_output(
                self._output_key,
                doc_text.model_dump(),
            )
            return result

        # pre-processing: apply gamma correction and tile image, as needed
        im_tiles = self._split_image(
            self._apply_gamma_correction(input.image), self.split_lim
        )
        logger.info(
            f"Image split into {len(im_tiles)} tiles. Extracting OCR text from each..."
        )

        ocr_blocks: List[Dict[str, Any]] = (
            []
        )  # list for OCR results across all tiles (whole image)
        for tile_num, tile in enumerate(im_tiles):
            logger.info(f"Processing tile {tile_num + 1} of {len(im_tiles)}")
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

        # write to cache
        self.write_result_to_cache(doc_text_extraction, doc_key)

        result = self._create_result(input)
        result.add_output(self._output_key, doc_text_extraction.model_dump())
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
