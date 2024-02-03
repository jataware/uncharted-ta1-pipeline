from pathlib import Path
from typing import List
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.common.pipeline import (
    Pipeline,
    BaseModelListOutput,
    BaseModelOutput,
    Output,
    OutputCreator,
    ImageOutput,
    PipelineResult,
)
from schema.ta1_schema import PageExtraction, ExtractionIdentifier, ProvenanceType
from PIL import ImageDraw, ImageFont
import tqdm


class TextExtractionPipeline(Pipeline):
    """
    Pipeline for extracting text from images using OCR.

    Args:
        work_dir (Path): The directory where OCR output will be saved.
        tile (bool): Whether to tile the image before OCR.
        pixel_limit (int): The maximum number of pixels in the image before resizing / tiling will apply.
        debug_images (bool): Whether to output debug images.

    Returns:
        List[DocTextExtraction]: A list of DocTextExtraction objects containing the extracted text.
    """

    def __init__(self, work_dir: Path, tile=True, pixel_limit=6000, debug_images=False):
        if tile:
            tasks = [
                ResizeTextExtractor("resize_text", work_dir, False, True, pixel_limit)
            ]
        else:
            tasks = [TileTextExtractor("tile_text", work_dir, pixel_limit)]

        outputs = [
            IntegrationOutput("integration_output"),
            DocTextExtractionOutput("doc_text_extraction_output"),
        ]
        if debug_images:
            outputs.append(OCRImageOutput("ocr_image_output"))

        super().__init__(
            "text_extraction",
            "Text Extraction",
            outputs,
            tasks,
        )


class IntegrationOutput(OutputCreator):
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

        page_extractions: List[PageExtraction] = []
        for i, text_extraction in enumerate(doc_text_extraction.extractions):
            page_extraction = PageExtraction(
                name="ocr",
                model=ExtractionIdentifier(
                    id=i, model="google-cloud-vision", field="ocr"
                ),
                ocr_text=text_extraction.text,
                bounds=[(v.x, v.y) for v in text_extraction.bounds],
                color_estimation=None,
                confidence=text_extraction.confidence,
                provenance=ProvenanceType.modelled,
            )
            page_extractions.append(page_extraction)

        result = BaseModelListOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, page_extractions
        )
        return result


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
        font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
        for text in tqdm.tqdm(extracted_text.extractions):
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
