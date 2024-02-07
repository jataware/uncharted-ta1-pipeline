import os
from pathlib import Path
from PIL import ImageDraw
from tasks.metadata_extraction.metadata_extraction import MetadataExtractor, LLM
from tasks.metadata_extraction.text_filter import TextFilter, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.entities import MapSegmentation
from tasks.text_extraction.entities import DocTextExtraction
from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.segmentation.detectron_segmenter import (
    DetectronSegmenter,
    SEGMENTATION_OUTPUT_KEY,
)
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
    ImageOutput,
    GeopackageOutput,
)
from schema.ta1_schema import (
    Map,
    MapFeatureExtractions,
    MapMetadata,
    GeoReferenceMeta,
    ProvenanceType,
)
from criticalmaas.ta1_geopackage import GeopackageDatabase


class MetadataExtractorPipeline(Pipeline):
    def __init__(
        self,
        work_dir: str,
        model_data_path: str,
        debug_images=False,
        model=LLM.GPT_3_5_TURBO,
    ):
        # extract text from image, filter out the legend and map areas, and then extract metadata using an LLM
        tasks = [
            ResizeTextExtractor(
                "resize_text", Path(work_dir).joinpath("text"), False, True, 6000
            ),
            DetectronSegmenter(
                "detectron_segmenter",
                model_data_path,
                str(Path(work_dir).joinpath("segmentation")),
            ),
            TextFilter(
                "text_filter",
                classes=[
                    "cross_section",
                    "legend_points_lines",
                    "legend_polygons",
                ],
            ),
            MetadataExtractor("metadata_extractor", model=model),
        ]

        outputs = [
            MetadataExtractionOutput("metadata_extraction_output"),
            IntegrationOutput("metadata_integration_output"),
        ]
        if debug_images:
            outputs.append(FilteredOCROutput("filtered_ocr_output"))

        super().__init__("metadata_extraction", "Metadata Extraction", outputs, tasks)

        self._ocr_output = Path(os.path.join(work_dir, "ocr_output"))


class MetadataExtractionOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a MetadataExtraction object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MetadataExtraction: The metadata extraction object.
        """
        metadata_extraction = MetadataExtraction.model_validate(
            pipeline_result.data[METADATA_EXTRACTION_OUTPUT_KEY]
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            metadata_extraction,
        )


class IntegrationOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a TA1 schema Map object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Map: The TA1 schema Map object.
        """
        metadata_extraction = MetadataExtraction.model_validate(
            pipeline_result.data[METADATA_EXTRACTION_OUTPUT_KEY]
        )

        schema_metadata = MapMetadata(
            id=metadata_extraction.map_id,
            authors=", ".join(metadata_extraction.authors),
            publisher="",
            year=(
                int(metadata_extraction.year)
                if metadata_extraction.year.isdigit()
                else -1
            ),
            organization="",
            scale=metadata_extraction.scale,
            confidence=None,  # TODO -- put in metadata extraction confidence?
            provenance=ProvenanceType.modelled,
        )

        schema_map = Map(
            name=metadata_extraction.title,
            id=metadata_extraction.map_id,
            source_url="",
            image_url="",
            image_size=[],
            map_metadata=schema_metadata,
            features=MapFeatureExtractions(
                lines=[], points=[], polygons=[], pipelines=[]
            ),
            cross_sections=None,
            projection_info=GeoReferenceMeta(
                gcps=[], projection="", bounds=None, provenance=None
            ),
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, schema_map
        )


class FilteredOCROutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a filtered OCR output image from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            ImageOutput: An image showing the text that passed through the filtering process.
        """
        filtered_text = DocTextExtraction.model_validate(
            pipeline_result.data[TEXT_EXTRACTION_OUTPUT_KEY]
        )
        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        text_image = pipeline_result.image.copy()
        draw = ImageDraw.Draw(text_image)
        # draw in the text bounds
        for text in filtered_text.extractions:
            # create a copy of the image
            if pipeline_result.image is not None:
                points = [(point.x, point.y) for point in text.bounds]
                draw.polygon(
                    points,
                    outline="#ff497b",
                    width=10,
                )
        # draw in the map region bounds
        if SEGMENTATION_OUTPUT_KEY in pipeline_result.data:
            map_segmentation = MapSegmentation.model_validate(
                pipeline_result.data[SEGMENTATION_OUTPUT_KEY]
            )
            for segment in map_segmentation.segments:
                points = [(point[0], point[1]) for point in segment.poly_bounds]
                draw.polygon(
                    points,
                    outline="#5ec04a",
                    width=20,
                )
        return ImageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, text_image
        )


class GeopackageIntegrationOutput(OutputCreator):
    def __init__(self, id: str, file_path: Path):
        super().__init__(id)
        self._file_path = file_path

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a geopackage output from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            GeopackageOutput: A geopackage containing the metadata extraction results.
        """

        metadata_extraction = MetadataExtraction.model_validate(
            pipeline_result.data[METADATA_EXTRACTION_OUTPUT_KEY]
        )

        db = GeopackageDatabase(str(self._file_path), crs="EPSG:4326")

        if db.model is None:
            raise ValueError("db.model is None")

        db.write_models(
            [
                db.model.map(
                    id=pipeline_result.raster_id,
                    name=pipeline_result.raster_id,
                    source_url="",
                    image_url="",
                    image_width=(
                        pipeline_result.image.width if pipeline_result.image else -1
                    ),
                    image_height=(
                        pipeline_result.image.height if pipeline_result.image else -1
                    ),
                ),
                db.model.map_metadata(
                    id=metadata_extraction.map_id,
                    map_id=pipeline_result.raster_id,
                    provenance=ProvenanceType.modelled.value,
                    authors=",".join(metadata_extraction.authors),
                    publisher="",
                    confidence=0.5,
                    year=int(metadata_extraction.year),
                    scale=metadata_extraction.scale,
                    title=metadata_extraction.title,
                ),
            ]
        )
        return GeopackageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
        )
