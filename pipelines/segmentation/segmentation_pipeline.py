import logging
from pathlib import Path
from PIL import ImageDraw, ImageFont
from typing import Dict, List
from collections import defaultdict
from tasks.common.io import append_to_cache_location, get_file_source
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.common.pipeline import (
    EmptyOutput,
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
    ImageOutput,
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
        debug_images=False,
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
                append_to_cache_location(model_data_cache_path, "text"),
                False,
                True,
                6000,
                0.5,
            ),
            DetectronSegmenter(
                "segmenter",
                model_data_path,
                append_to_cache_location(model_data_cache_path, "segmentation"),
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
        if debug_images:
            outputs.append(DebugImagesOutput("map_segmentation_debug_images_output"))

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
        result = pipeline_result.data.get(SEGMENTATION_OUTPUT_KEY, None)
        if result is None:
            return EmptyOutput()

        map_segmentation = MapSegmentation.model_validate(result)
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
        result = pipeline_result.data.get(SEGMENTATION_OUTPUT_KEY, None)
        if result is None:
            return EmptyOutput()

        map_segmentation = MapSegmentation.model_validate(result)
        mapper = SegmentationMapper(MODEL_NAME, MODEL_VERSION)

        cdr_segmentation = mapper.map_to_cdr(map_segmentation)
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, cdr_segmentation
        )


class DebugImagesOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates an output image with segmentation areas overlayed

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            ImageOutput: An image showing with segmentation areas overlayed on the map
        """
        category2colour = {
            "cross_section": "limegreen",
            "legend_points_lines": "blue",
            "legend_polygons": "red",
            "map": "orange",
        }
        font_size = 32
        poly_line_width = 16
        font = ImageFont.load_default(font_size)

        result = pipeline_result.data.get(SEGMENTATION_OUTPUT_KEY, None)
        if result is None:
            return EmptyOutput()

        map_segmentation = MapSegmentation.model_validate(result)
        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        debug_image = pipeline_result.image.copy()
        draw = ImageDraw.Draw(debug_image)

        # group segments by class and draw polygons
        segs_per_class = defaultdict(list)
        for seg in map_segmentation.segments:
            segs_per_class[seg.class_label].append(seg)
            color = category2colour.get(seg.class_label, "yellow")
            xy = [(point[0], point[1]) for point in seg.poly_bounds]
            draw.polygon(xy, outline=color, width=poly_line_width)

        # draw the labels after (on top)
        for seg_class, segs in segs_per_class.items():
            rank = 1
            color = category2colour.get(seg_class, "yellow")
            for seg in segs:
                label = f"{seg_class} {rank}, conf: {seg.confidence:.3f}"
                # draw label in the top left
                text_xy = (
                    seg.bbox[0],
                    max(seg.bbox[1] - 1.2 * font_size, 0),
                )
                draw.text(
                    text_xy,
                    label,
                    fill=color,
                    font=font,
                    font_size=font_size,
                )
                rank += 1

        return ImageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, debug_image
        )
