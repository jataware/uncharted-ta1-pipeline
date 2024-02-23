from typing import List
from pathlib import Path
import datetime
from schema.ta1_schema import PageExtraction, ExtractionIdentifier, ProvenanceType
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.common.pipeline import (
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
    BaseModelListOutput,
    GeopackageOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from criticalmaas.ta1_geopackage import GeopackageDatabase
from shapely import Polygon
from geopandas import GeoDataFrame


class SegmentationPipeline(Pipeline):
    """
    Pipeline for segmenting maps into different components (map area, legend, etc.).
    """

    def __init__(
        self,
        model_data_path: str,
        model_data_cache_path: str = "",
        confidence_thres=0.25,
    ):
        """
        Initializes the pipeline.

        Args:
            config_path (str): The path to the Detectron2 config file.
            model_weights_path (str): The path to the Detectron2 model weights file.
            confidence_thres (float): The confidence threshold for the segmentation.
        """

        tasks = [
            DetectronSegmenter(
                "segmenter",
                model_data_path,
                model_data_cache_path,
                confidence_thres=confidence_thres,
            )
        ]

        outputs = [
            MapSegmentationOutput("map_segmentation_output"),
            IntegrationOutput("integration_output"),
        ]
        super().__init__("map-segmentation", "Map Segmentation", outputs, tasks)


class MapSegmentationOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult):
        """
        Creates a MapSegmentation object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MapSegmentation: The map segmentation extraction object.
        """
        map_segmentation = MapSegmentation.model_validate(
            pipeline_result.data[SEGMENTATION_OUTPUT_KEY]
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_segmentation,
        )


class IntegrationOutput(OutputCreator):
    """
    OutputCreator for text extraction pipeline.
    """

    def __init__(self, id):
        """
        Initializes the output creator.

        Args:
            id (str): The ID of the output creator.
        """
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Validates the pipeline result and converts into the TA1 schema representation

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_segmentation = MapSegmentation.model_validate(
            pipeline_result.data[SEGMENTATION_OUTPUT_KEY]
        )

        page_extractions: List[PageExtraction] = []
        for i, segment in enumerate(map_segmentation.segments):
            page_extraction = PageExtraction(
                name="segmentation",
                model=ExtractionIdentifier(
                    id=i, model=segment.id_model, field=segment.class_label
                ),
                ocr_text="",
                bounds=segment.poly_bounds,
                color_estimation=None,
                confidence=segment.confidence,
                provenance=ProvenanceType.modelled,
            )
            page_extractions.append(page_extraction)

        result = BaseModelListOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, page_extractions
        )
        return result


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
            GeopackageOutput: A geopackage containing the segmentation results.
        """
        map_segmentation = MapSegmentation.model_validate(
            pipeline_result.data[SEGMENTATION_OUTPUT_KEY]
        )

        db = GeopackageDatabase(str(self._file_path), crs="EPSG:4326")

        if db.model is None:
            raise ValueError("db.model is None")

        if not map_segmentation.segments:
            # no segmentation results for this image
            return GeopackageOutput(
                pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
            )

        # TODO: these db write calls will throw an exception if records already exist. Ideally should check if records exist or overwrite?

        model_ver = map_segmentation.segments[0].id_model
        model_run_id = f"segmentation_{model_ver}_{pipeline_result.raster_id}"

        db.write_models(
            [
                # map
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
                # model run
                db.model.model_run(
                    id=model_run_id,
                    model_name="segmentation",
                    version=model_ver,
                    timestamp=str(datetime.datetime.now()),
                    map_id=pipeline_result.raster_id,
                ),
            ]
        )

        extraction_identifiers = []
        page_extractions = []
        for i, segment in enumerate(map_segmentation.segments):

            extr_id = f"{pipeline_result.raster_id}_{segment.id_model}_{i}"
            poly_obj = Polygon(segment.poly_bounds)

            # identifier (pointer) for this extraction
            # (NOTE: using extraction ID = <raster_id>_<seg id_model>_<i>
            extr_id = f"{pipeline_result.raster_id}_{segment.id_model}_{i}"
            extraction_identifiers.append(
                db.model.extraction_pointer(
                    id=extr_id,
                    # TODO - geo_package "table_name" enum is missing types of extractions (ocr, segmentation, etc)
                    # use "map_metadata" here for now
                    table_name="map_metadata",
                    column_name=segment.class_label,  # field name of the model
                    record_id=extr_id,  # ID of the extracted feature
                )
            )

            page_extr = dict(
                id=extr_id,
                name="segmentation",
                pointer=extr_id,
                model_run=model_run_id,
                ocr_text=None,
                color_estimation=None,
                px_geometry=poly_obj,
                bounds="",
                confidence=segment.confidence,
                provenance=ProvenanceType.modelled.value,  # type: ignore
            )
            page_extractions.append(page_extr)

        if extraction_identifiers:
            db.write_models(extraction_identifiers)
        if page_extractions:
            db.write_dataframe(GeoDataFrame(page_extractions, geometry="px_geometry"), "page_extraction")  # type: ignore

        return GeopackageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
        )
