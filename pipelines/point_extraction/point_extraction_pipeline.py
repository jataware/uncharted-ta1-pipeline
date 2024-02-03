import logging
from pathlib import Path
from typing import List
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    BaseModelListOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
)
from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.segmentation.detectron_segmenter import (
    DetectronSegmenter,
    SEGMENTATION_OUTPUT_KEY,
)
from schema.ta1_schema import (
    PointFeature,
    PointType,
    ProvenanceType,
)


logger = logging.getLogger(__name__)


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
            # TODO: could use Tiled Text extractor here? ... better recall for dip magnitude extraction?
            ResizeTextExtractor(
                "resize_text",
                Path(work_dir).joinpath("text"),
                to_blocks=True,
                document_ocr=False,
                pixel_lim=6000,
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
                    "point_detection", model_path, work_dir, batch_size=20
                ),
                Untiler("untiling"),
                PointOrientationExtractor("point_orientation_extraction"),
            ]
        )

        outputs = [
            MapPointLabelOutput("map_point_label_output"),
            IntegrationOutput("integration_output"),
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


class IntegrationOutput(OutputCreator):
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

        point_features: List[PointFeature] = []
        if map_image.labels:
            for i, map_pt_label in enumerate(map_image.labels):

                # TODO - add in optional point class description text? (eg extracted from legend?)
                pt_type = PointType(
                    id=str(map_pt_label.class_id),
                    name=map_pt_label.class_name,
                    description=None,
                )
                # centre pixel of point bounding box
                xy_centre = (
                    int((map_pt_label.x2 + map_pt_label.x1) / 2),
                    int((map_pt_label.y2 + map_pt_label.y1) / 2),
                )

                pt_feature = PointFeature(
                    id=str(i),  # TODO - use a better unique ID here?
                    type=pt_type,
                    map_geom=None,  # TODO - add in real-world coords if/when geo-ref transform available
                    px_geom=xy_centre,
                    dip_direction=map_pt_label.direction,
                    dip=map_pt_label.dip,
                    confidence=map_pt_label.score,
                    provenance=ProvenanceType.modelled,
                )
                point_features.append(pt_feature)

        result = BaseModelListOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, point_features
        )
        return result
