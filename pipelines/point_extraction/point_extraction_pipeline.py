import logging
from pathlib import Path
from typing import Dict, List

from matplotlib.dates import MO
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.features.point_features import (
    PointFeatureCollection,
    PointLegendAndFeaturesResult,
    PointFeature,
    Point,
    PointProperties,
)
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.text_extraction.text_extractor import TileTextExtractor
from tasks.segmentation.detectron_segmenter import (
    DetectronSegmenter,
    SEGMENTATION_OUTPUT_KEY,
)


logger = logging.getLogger(__name__)

import importlib.metadata

MODEL_NAME = "lara-point-extraction"  # should match name in pyproject.toml
MODEL_VERSION = importlib.metadata.version(MODEL_NAME)


class PointExtractionPipeline(Pipeline):
    """
    Pipeline for extracting map point symbols, orientation, and their associated orientation and magnitude values, if present

    Args:
        model_path: path to point symbol extraction model weights
        model_path_segmenter: path to segmenter model weights
        work_dir: cache directory
    """

    def __init__(
        self,
        model_path: str,
        model_path_segmenter: str,
        work_dir: str,
        verbose=False,
    ):
        # extract text from image, segmentation to only keep the map area,
        # tile, extract points, untile, predict direction
        logger.info("Initializing Point Extraction Pipeline")
        tasks = []
        tasks.append(
            TileTextExtractor(
                "tile_text",
                Path(work_dir).joinpath("text"),
            )
        )
        if model_path_segmenter:
            tasks.append(
                DetectronSegmenter(
                    "detectron_segmenter",
                    model_path_segmenter,
                    str(Path(work_dir).joinpath("segmentation")),
                )
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.extend(
            [
                Tiler("tiling"),
                YOLOPointDetector(
                    "point_detection",
                    model_path,
                    str(Path(work_dir).joinpath("points")),
                    batch_size=20,
                ),
                Untiler("untiling"),
                PointOrientationExtractor("point_orientation_extraction"),
            ]
        )

        outputs = [
            MapPointLabelOutput("map_point_label_output"),
            CDROutput("map_point_label_cdr_output"),
        ]

        super().__init__("point_extraction", "Point Extraction", outputs, tasks)
        self._verbose = verbose


class MapPointLabelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a MapPointLabel object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MapPointLabel: The map point label extraction object.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_image,
        )


class CDROutput(OutputCreator):
    """
    OutputCreator for point extraction pipeline.
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
        Validates the point extraction pipeline result and converts into the TA1 schema representation

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])

        point_features: List[PointLegendAndFeaturesResult] = []

        # create seperate lists for each point class since they are groupded by class
        # in the results
        point_features_by_class: Dict[str, List[PointFeature]] = {}
        point_id = 0
        if map_image.labels:
            for map_pt_label in map_image.labels:
                if map_pt_label.class_name not in point_features_by_class:
                    point_features_by_class[map_pt_label.class_name] = []

                # create the point geometry
                point = Point(
                    coordinates=[
                        (map_pt_label.x1 + map_pt_label.x2) / 2,
                        (map_pt_label.y1 + map_pt_label.y2) / 2,
                    ]
                )

                # create the additional point properties
                properties = PointProperties(
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                    confidence=map_pt_label.score,
                    bbox=[
                        map_pt_label.x1,
                        map_pt_label.y1,
                        map_pt_label.x2,
                        map_pt_label.y2,
                    ],
                    dip=round(map_pt_label.dip) if map_pt_label.dip else None,
                    dip_direction=(
                        round(map_pt_label.direction)
                        if map_pt_label.direction
                        else None
                    ),
                )

                # add the point geometry and properties to the point feature
                point_feature = PointFeature(
                    id=f"{map_pt_label.class_id}.{point_id}",
                    geometry=point,
                    properties=properties,
                )
                point_id += 1

                # add to the list of point features for the class
                point_features_by_class[map_pt_label.class_name].append(point_feature)

        # create a PointLegendAndFeaturesResult for each class - we don't use the legend
        # so the legend bbox is empty
        for class_name, features in point_features_by_class.items():
            point_features_result = PointLegendAndFeaturesResult(
                id="id",
                crs="CRITICALMAAS:pixel",
                cdr_projection_id=None,
                name=class_name,
                description=None,
                legend_bbox=None,
                point_features=[PointFeatureCollection(features=features)],
            )

        # add to our final list of features results and create the output
        point_features.append(point_features_result)

        feature_results = FeatureResults(
            cog_id=pipeline_result.raster_id,
            line_feature_results=None,
            point_feature_results=point_features,
            polygon_feature_results=None,
            cog_area_extractions=None,
            cog_metadata_extractions=None,
            system=MODEL_NAME,
            system_version=MODEL_VERSION,
        )

        result = BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, feature_results
        )
        return result
