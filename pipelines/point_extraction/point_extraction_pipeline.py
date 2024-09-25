import logging
from pathlib import Path
from typing import List

from flask import app

from schema.mappers.cdr import PointsMapper
from tasks.common.io import append_to_cache_location
from tasks.point_extraction.legend_analyzer import (
    LegendPreprocessor,
    LegendPostprocessor,
)
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.point_extractor_utils import convert_preds_to_bitmasks
from tasks.point_extraction.template_match_point_extractor import (
    TemplateMatchPointExtractor,
)
from tasks.point_extraction.finalize_point_extractions import FinalizePointExtractions
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import (
    PointLabels,
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
    MAP_PT_LABELS_OUTPUT_KEY,
)
from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
    ImageDictOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.segmentation.denoise_segments import DenoiseSegments
from tasks.text_extraction.text_extractor import TileTextExtractor


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
        cache_location: str,
        verbose=False,
        fetch_legend_items=False,
        include_cdr_output=True,
        include_bitmasks_output=False,
        gpu=True,
        batch_size=20,
    ):
        # extract text from image, segmentation to only keep the map area,
        # tile, extract points, untile, predict direction
        logger.info("Initializing Point Extraction Pipeline")
        yolo_point_extractor = YOLOPointDetector(
            "point_detection",
            model_path,
            append_to_cache_location(cache_location, "points"),
            batch_size=batch_size,
            device="auto" if gpu else "cpu",
        )

        tasks = []
        tasks.append(
            TileTextExtractor(
                "tile_text",
                append_to_cache_location(cache_location, "text"),
                gamma_correction=0.5,
            )
        )
        if model_path_segmenter:
            tasks.extend(
                [
                    DetectronSegmenter(
                        "segmenter",
                        model_path_segmenter,
                        append_to_cache_location(cache_location, "segmentation"),
                        gpu=gpu,
                    ),
                    DenoiseSegments("segment_denoising"),
                ]
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.extend(
            [
                LegendPreprocessor("legend_preprocessor", "", fetch_legend_items),
                Tiler("tiling"),
                yolo_point_extractor,
                Untiler("untiling"),
                PointOrientationExtractor(
                    "point_orientation_extraction",
                    yolo_point_extractor._model_id,
                    append_to_cache_location(cache_location, "point_orientations"),
                ),
                LegendPostprocessor("legend_postprocessor", ""),
                TemplateMatchPointExtractor(
                    "template_match_point_extraction",
                    append_to_cache_location(cache_location, "template_match_points"),
                ),
                FinalizePointExtractions("finalize_points"),
            ]
        )

        outputs: List[OutputCreator] = [
            MapPointLabelOutput("map_point_label_output"),
        ]
        if include_cdr_output:
            outputs.append(CDROutput("map_point_label_cdr_output"))
        if include_bitmasks_output:
            outputs.append(BitmasksOutput("map_point_label_bitmasks_output"))

        super().__init__("point_extraction", "Point Extraction", outputs, tasks)
        self._verbose = verbose


class MapPointLabelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a PointLabel object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            PointLabel: The map point label extraction object.
        """
        map_point_labels = PointLabels.model_validate(
            pipeline_result.data[MAP_PT_LABELS_OUTPUT_KEY]
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_point_labels,
        )


class CDROutput(OutputCreator):
    """
    Create CDR output objects for point extraction pipeline.
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
        map_point_labels = PointLabels.model_validate(
            pipeline_result.data[MAP_PT_LABELS_OUTPUT_KEY]
        )

        mapper = PointsMapper(MODEL_NAME, MODEL_VERSION)

        cdr_points = mapper.map_to_cdr(map_point_labels)
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, cdr_points
        )


class BitmasksOutput(OutputCreator):
    """
    Create bitmasks output (in CMA-contest format) for point extraction pipeline.
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
        Validates the point extraction pipeline result and converts to bitmasks

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_point_labels = PointLabels.model_validate(
            pipeline_result.data[MAP_PT_LABELS_OUTPUT_KEY]
        )
        legend_pt_items = LegendPointItems(items=[])
        if LEGEND_ITEMS_OUTPUT_KEY in pipeline_result.data:
            legend_pt_items = LegendPointItems.model_validate(
                pipeline_result.data[LEGEND_ITEMS_OUTPUT_KEY]
            )

        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        (w, h) = pipeline_result.image.size
        bitmasks_dict = convert_preds_to_bitmasks(
            map_point_labels, legend_pt_items, (w, h)
        )

        return ImageDictOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, bitmasks_dict
        )
