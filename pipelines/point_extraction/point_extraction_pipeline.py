import logging
from pathlib import Path
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointDirectionPredictor
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
)

from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.segmentation.detectron_segmenter import (
    DetectronSegmenter,
    SEGMENTATION_OUTPUT_KEY,
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
                YOLOPointDetector("point_detection", model_path, work_dir),
                Untiler("untiling"),
                PointDirectionPredictor("point_direction_prediction"),
            ]
        )

        outputs = [
            MapPointLabelOutput("map_point_label_output"),
        ]

        super().__init__("point_extraction", "Point Extraction", outputs, tasks)
        self._verbose = verbose


class MapPointLabelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult):
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
