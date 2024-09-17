import logging
from pathlib import Path
from typing import Dict, List
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.common.pipeline import (
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
)
from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.segmentation.text_with_segments import TextWithSegments
from schema.mappers.cdr import SegmentationMapper

logger = logging.getLogger("segmentation_pipeline")

import importlib.metadata

MODEL_NAME = "lara-map-segmentation"  # should match name in pyproject.toml
MODEL_VERSION = importlib.metadata.version(MODEL_NAME)


class SegmentationPipeline(Pipeline):
    """
    Pipeline for segmenting maps into different components (map area, legend, etc.).
    """

    def __init__(
        self,
        model_data_path: str,
        model_data_cache_path: str = "",
        cdr_schema=False,
        confidence_thres=0.25,
        gpu=True,
    ):
        """
        Initializes the pipeline.

        Args:
            config_path (str): The path to the Detectron2 config file.
            model_weights_path (str): The path to the Detectron2 model weights file.
            confidence_thres (float): The confidence threshold for the segmentation.
        """

        tasks = [
            ResizeTextExtractor(
                "resize_text",
                Path(model_data_cache_path).joinpath("text"),
                False,
                True,
                6000,
                0.5,
            ),
            DetectronSegmenter(
                "segmenter",
                model_data_path,
                str(
                    Path(
                        model_data_cache_path,
                    ).joinpath("segmentation")
                ),
                confidence_thres=confidence_thres,
                gpu=gpu,
            ),
            TextWithSegments("text_with_segments"),
        ]

        outputs: List[OutputCreator] = [
            MapSegmentationOutput("map_segmentation_output")
        ]
        if cdr_schema:
            outputs.append(CDROutput("map_segmentation_cdr_output"))

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
        mapper = SegmentationMapper(MODEL_NAME, MODEL_VERSION)

        cdr_segmentation = mapper.map_to_cdr(map_segmentation)
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, cdr_segmentation
        )
