import os
from pathlib import Path
from tasks.metadata_extraction.metadata_extraction import (
    MetadataExtractor,
)
from tasks.metadata_extraction.entities import MetadataExtraction
from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
)
from schema.ta1_schema import Map, MapFeatureExtractions, ProjectionMeta


class MetadataExtractorPipeline(Pipeline):
    def __init__(
        self,
        work_dir: Path,
        verbose=False,
    ):
        # extract text from image and then extract metadata using an LLM
        tasks = [
            ResizeTextExtractor("resize_text", work_dir, False, True, 6000),
            MetadataExtractor("metadata_extractor", verbose),
        ]

        outputs = [
            MetadataExtractionOutput("metadata_extraction_output"),
            IntegrationOutput("metadata_integration_output"),
        ]

        super().__init__("metadata_extraction", "Metadata Extraction", outputs, tasks)

        self._ocr_output = Path(os.path.join(work_dir, "ocr_output"))
        self._verbose = verbose


class MetadataExtractionOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult):
        """
        Creates a MetadataExtraction object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MetadataExtraction: The metadata extraction object.
        """
        metadata_extraction = MetadataExtraction.model_validate(pipeline_result.data)
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            metadata_extraction,
        )


class IntegrationOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult):
        metadata_extraction = MetadataExtraction.model_validate(pipeline_result.data)
        schema_map = Map(
            name=metadata_extraction.title,
            source_url="",
            image_url="",
            image_size=[],
            authors=", ".join(metadata_extraction.authors),
            publisher="",
            year=int(metadata_extraction.year)
            if metadata_extraction.year.isdigit()
            else -1,
            organization="",
            scale=metadata_extraction.scale,
            bounds="",
            features=MapFeatureExtractions(lines=[], points=[], polygons=[]),
            cross_sections=None,
            pipelines=[],
            projection_info=ProjectionMeta(gcps=[], projection=""),
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, schema_map
        )
