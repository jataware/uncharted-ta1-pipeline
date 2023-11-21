from pathlib import Path
from typing import List
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor
from tasks.text_extraction.entities import DocTextExtraction
from tasks.common.pipeline import (
    Pipeline,
    BaseModelListOutput,
    BaseModelOutput,
    Output,
    OutputCreator,
    PipelineResult,
)
from schema.ta1_schema import PageExtraction, ExtractionIdentifier


class TextExtractionPipeline(Pipeline):
    """
    Pipeline for extracting text from images using OCR.

    Args:
        work_dir (Path): The directory where OCR output will be saved.
        tile (bool): Whether to tile the image before OCR.
        pixel_limit (int): The maximum number of pixels in the image before resizing / tiling will apply.

    Returns:
        List[DocTextExtraction]: A list of DocTextExtraction objects containing the extracted text.
    """

    def __init__(self, work_dir: Path, tile=True, pixel_limit=6000):
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
        doc_text_extraction = DocTextExtraction.model_validate(pipeline_result.data)

        page_extractions: List[PageExtraction] = []
        for text_extraction in doc_text_extraction.extractions:
            page_extraction = PageExtraction(
                name="ocr",
                model=ExtractionIdentifier(
                    id=1, model="google-cloud-vision", field="ocr"
                ),
                ocr_text=text_extraction.text,
                # confidence=text_extraction.confidence,
                bounds=[(v.x, v.y) for v in text_extraction.bounds],
                color_estimation=None,
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
        doc_text_extraction = DocTextExtraction.model_validate(pipeline_result.data)
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            doc_text_extraction,
        )
