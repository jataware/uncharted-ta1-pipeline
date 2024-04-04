import logging
from typing import Dict, List
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.common.pipeline import (
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from schema.cdr_schemas.area_extraction import Area_Extraction, AreaType
from schema.cdr_schemas.feature_results import FeatureResults

logger = logging.getLogger("segmentation_pipeline")


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
            CDROutput("map_segmentation_cdr_output"),
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


class CDROutput(OutputCreator):
    """
    CDR OutputCreator for map segmentation pipeline.
    """

    AREA_MAPPING = {
        "cross_section": AreaType.CrossSection,
        "legend_points_lines": AreaType.Line_Point_Legend_Area,
        "legend_polygons": AreaType.Polygon_Legend_Area,
        "map": AreaType.Map_Area,
    }

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

        area_extractions: List[Area_Extraction] = []
        # create CDR area extractions for segment we've identified in the map
        for i, segment in enumerate(map_segmentation.segments):
            coordinates = [list(point) for point in segment.poly_bounds]

            if segment.class_label in CDROutput.AREA_MAPPING:
                area_type = CDROutput.AREA_MAPPING[segment.class_label]
            else:
                logger.warning(
                    f"Unknown area type {segment.class_label} for map {pipeline_result.raster_id}"
                )
                area_type = AreaType.Map_Area

            area_extraction = Area_Extraction(
                coordinates=[coordinates],
                bbox=segment.bbox,
                category=area_type,
                confidence=segment.confidence,  # assume two points - ll, ur
                model="",
                model_version="",
                text=None,
            )
            area_extractions.append(area_extraction)

        feature_results = FeatureResults(
            # relevant to segment extractions
            cog_id=pipeline_result.raster_id,
            cog_area_extractions=area_extractions,
            system="map-segmentation",
            system_version="1.0",
            # other
            line_feature_results=None,
            point_feature_results=None,
            polygon_feature_results=None,
            cog_metadata_extractions=None,
        )

        result = BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, feature_results
        )
        return result
