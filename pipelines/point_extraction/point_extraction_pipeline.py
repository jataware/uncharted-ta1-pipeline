from pathlib import Path
from tasks.point_extraction.point_extractor import PointDirectionPredictor, YOLOPointDetector
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
)
from schema.ta1_schema import Map, MapFeatureExtractions, ProjectionMeta


class PointExtractionPipeline(Pipeline):
    """
    Pipeline for extracting map points, orientation, and their associated incline values.
    """
    def __init__(
        self,
        model_data: Path,
        model_data_cache: Path,
        verbose=False,
    ):
        # tile, extract points, untile, predict direction
        tasks = [
            Tiler("tiling"),
            YOLOPointDetector("point_detection", str(model_data), str(model_data_cache)),
            Untiler("untiling"),
            PointDirectionPredictor("point_direction_prediction")
        ]

        outputs = [
            MapPointLabelOutput("map_point_label_output"),
        ]

        super().__init__("metadata_extraction", "Metadata Extraction", outputs, tasks)
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