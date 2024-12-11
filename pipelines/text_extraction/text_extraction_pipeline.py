from pathlib import Path
from typing import List
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.area_extraction import Area_Extraction, AreaType
from tasks.common.io import append_to_cache_location
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.common.pipeline import (
    Pipeline,
    BaseModelOutput,
    Output,
    OutputCreator,
    ImageOutput,
    PipelineResult,
)
from PIL import ImageDraw, ImageFont

import importlib.metadata

MODEL_NAME = "lara-text-extraction"  # should match name in pyproject.toml
MODEL_VERSION = importlib.metadata.version(MODEL_NAME)


class TextExtractionPipeline(Pipeline):
    """
    Pipeline for extracting text from images using OCR.

    Args:
        work_dir (Path): The directory where OCR output will be saved.
        tile (bool): Whether to tile the image before OCR.
        pixel_limit (int): The maximum number of pixels in the image before resizing / tiling will apply.
        gamma_corr (float): Image gamma correction prior to OCR. 1= no change (disabled); <1 lightens the image; >1 darkens the image
                            NOTE: gamma = 0.5 recommended for OCR pre-processing
        debug_images (bool): Whether to output debug images.

    Returns:
        List[DocTextExtraction]: A list of DocTextExtraction objects containing the extracted text.
    """

    def __init__(
        self,
        cache_location: str,
        tile=True,
        pixel_limit=6000,
        gamma_corr=1.0,
        debug_images=False,
        metrics_url: str = "",
    ):
        if tile:
            tasks = [
                TileTextExtractor(
                    "tile_text",
                    append_to_cache_location(cache_location, "text"),
                    pixel_limit,
                    gamma_corr,
                    metrics_url=metrics_url,
                )
            ]
        else:
            tasks = [
                ResizeTextExtractor(
                    "resize_text",
                    append_to_cache_location(cache_location, "text"),
                    False,
                    True,
                    pixel_limit,
                    gamma_corr,
                    metrics_url=metrics_url,
                )
            ]

        outputs: List[OutputCreator] = [
            DocTextExtractionOutput("doc_text_extraction_output"),
            CDROutput("doc_text_extraction_cdr_output"),
        ]
        if debug_images:
            outputs.append(OCRImageOutput("ocr_image_output"))

        super().__init__(
            "text_extraction",
            "Text Extraction",
            outputs,
            tasks,
        )


class DocTextExtractionOutput(OutputCreator):
    """
    OutputCreator for text extraction pipeline.

    Args:
        id (str): The ID of the output creator.

    Returns:
        Output: The output of the pipeline.
    """

    def __init__(self, id):
        """Initializes the output creator."""
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """Validates the pipeline result and converts into the TA1 schema representation"""
        doc_text_extraction = DocTextExtraction.model_validate(
            pipeline_result.data[TEXT_EXTRACTION_OUTPUT_KEY]
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            doc_text_extraction,
        )


class CDROutput(OutputCreator):
    def __init__(self, id):
        """Initializes the output creator."""
        super().__init__(id)

    @staticmethod
    def _get_bounding_box(text) -> List[float | int]:
        """
        Get the bounding box of the text.

        Args:
            text (TextExtraction): The text extraction.

        Returns:
            List[Union[float, int]]: The bounding box of the text expressed as llx, lly, urx, ury coordintes.
        """
        return [text.bounds[0].x, text.bounds[0].y, text.bounds[2].x, text.bounds[2].y]

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a CDR output from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline in the CDR schema format.
        """
        doc_text_extraction = DocTextExtraction.model_validate(
            pipeline_result.data[TEXT_EXTRACTION_OUTPUT_KEY]
        )

        area_extractions: List[Area_Extraction] = []
        # create CDR area extractions for segment we've identified in the map
        for text in doc_text_extraction.extractions:
            area_extractions.append(
                Area_Extraction(
                    category=AreaType.OCR,
                    coordinates=[[[point.x, point.y] for point in text.bounds]],
                    bbox=CDROutput._get_bounding_box(text),
                    text=text.text,
                    confidence=text.confidence,
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                )
            )

        feature_results = FeatureResults(
            cog_id=doc_text_extraction.doc_id,
            cog_area_extractions=area_extractions,
            system=MODEL_NAME,
            system_version=MODEL_VERSION,
        )

        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, feature_results
        )


class OCRImageOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates an OCR output image from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            ImageOutput: An image showing the text that passed through the filtering process.
        """
        extracted_text = DocTextExtraction.model_validate(
            pipeline_result.data[TEXT_EXTRACTION_OUTPUT_KEY]
        )
        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        text_image = pipeline_result.image.copy()
        draw = ImageDraw.Draw(text_image)
        # draw in the text bounds
        font = ImageFont.load_default(30)
        for text in extracted_text.extractions:
            points = [(point.x, point.y) for point in text.bounds]
            draw.polygon(
                points,
                outline="#ff497b",
                width=1,
            )
            # draw the text and continue on any encoding exceptions
            try:
                draw.text(points[3], text.text, fill="#ff497b", font=font)
            except Exception as e:
                continue
        return ImageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, text_image
        )
