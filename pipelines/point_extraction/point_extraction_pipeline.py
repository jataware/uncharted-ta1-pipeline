import logging
import math
from collections import defaultdict
from typing import List
from PIL import ImageDraw, ImageFont
from schema.mappers.cdr import PointsMapper
from tasks.common.io import append_to_cache_location
from tasks.common.task import EvaluateHalt
from tasks.segmentation.segmenter_utils import map_missing
from tasks.point_extraction.legend_analyzer import (
    LegendPreprocessor,
    LegendPostprocessor,
)
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.point_extractor_utils import (
    convert_preds_to_bitmasks,
    rotated_about,
)
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
    EmptyOutput,
    BaseModelOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
    ImageDictOutput,
    ImageOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.segmentation.denoise_segments import DenoiseSegments
from tasks.segmentation.segmenter_utils import map_missing
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
        metrics_url="",
        debug_images=False,
        ocr_cloud_auth=False,
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
        if model_path_segmenter:
            tasks.extend(
                [
                    DetectronSegmenter(
                        "segmenter",
                        model_path_segmenter,
                        append_to_cache_location(cache_location, "segmentation"),
                        gpu=gpu,
                    ),
                    # early termination if no map region is found
                    EvaluateHalt("map_presence_check", map_missing),
                    DenoiseSegments("segment_denoising"),
                ]
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.extend(
            [
                TileTextExtractor(
                    "tile_text",
                    append_to_cache_location(cache_location, "text"),
                    gamma_correction=0.5,
                    metrics_url=metrics_url,
                    ocr_cloud_auth=ocr_cloud_auth,
                ),
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
        if debug_images:
            outputs.append(DebugImagesOutput("map_point_label_debug_images_output"))

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


class DebugImagesOutput(OutputCreator):
    def __init__(self, id: str, draw_rotated_rectangles: bool = True):
        self._draw_rotated_rectangles = draw_rotated_rectangles
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates an output image with point extraction bboxes overlayed

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            ImageOutput: An image showing with point extraction bboxes overlayed on the map
        """
        logger.info(
            f"Creating point extraction viz image for raster: {pipeline_result.raster_id}..."
        )

        class2colour = {
            "strike_and_dip": "red",
            "horizontal_bedding": "blue",
            "overturned_bedding": "green",
            "vertical_bedding": "orange",
            "inclined_foliation": "darkmagenta",
            "inclined_foliation_igneous": "limegreen",
            "vertical_foliation": "springgreen",
            "vertical_joint": "turquoise",
            "gravel_borrow_pit": "hotpink",
            "mine_shaft": "darkviolet",
            "prospect": "deepskyblue",
            "mine_tunnel": "tomato",
            "mine_quarry": "limegreen",
            "sink_hole": "goldenrod",
            "lineation": "violet",
            "drill_hole": "cyan",
        }
        font_size = 16
        line_width = 4
        font = ImageFont.load_default(font_size)

        result = pipeline_result.data.get(MAP_PT_LABELS_OUTPUT_KEY, None)
        if result is None:
            return EmptyOutput()
        map_point_labels = PointLabels.model_validate(result)

        if pipeline_result.image is None:
            raise ValueError("Pipeline result image is None")
        debug_image = pipeline_result.image.copy()
        draw = ImageDraw.Draw(debug_image, mode="RGBA")

        if not map_point_labels.labels:
            return EmptyOutput()

        counts_per_class = defaultdict(int)
        for label in map_point_labels.labels:

            box_coords = (label.x1, label.y1, label.x2, label.y2)
            class_name = label.class_name
            # score = label.score
            if class_name not in class2colour:
                logger.warning(
                    f"Point class_name not recognized: {class_name}. Plotting with yellow"
                )
                colour = "yellow"
            else:
                colour = class2colour[class_name]
            if self._draw_rotated_rectangles and label.direction:
                rot_angle_compass = float(label.direction)
                # convert angle compass angle convention (CW with 0 deg at top)
                # regular 'trig' angle convention (CCW with 0 to the right)
                rot_angle = 270 - rot_angle_compass
                if rot_angle < 0:
                    rot_angle += 360
                xc = (label.x1 + label.x2) / 2.0
                yc = (label.y1 + label.y2) / 2.0
                square_vertices = [
                    (label.x1, label.y1),
                    (label.x2, label.y1),
                    (label.x2, label.y2),
                    (label.x1, label.y2),
                ]
                square_vertices = [
                    rotated_about(x, y, xc, yc, math.radians(-rot_angle))
                    for x, y in square_vertices
                ]
                draw.polygon(square_vertices, outline=colour, width=line_width)
            else:
                draw.rectangle(box_coords, outline=colour, width=line_width)

            class_text = class_name.replace("_", " ")
            draw.text(
                (label.x1, label.y1 - font_size),
                class_text,
                fill=colour,
                font=font,
                font_size=font_size,
            )

            angles_text = ""
            # don't write strike angles for now (shown by rotated rect)
            # if label["direction"]:
            #    angles_text = f'angle: {int(label["direction"])}'
            if label.dip:
                if angles_text:
                    angles_text += ", "
                angles_text += f"dip: {int(label.dip)}"
            if angles_text:
                draw.text(
                    (label.x1, label.y2),
                    angles_text,
                    fill=colour,
                    font=font,
                    font_size=font_size,
                )

            counts_per_class[class_name] += 1

        print("--------")
        print(f"Point Extractions per class for raster {pipeline_result.raster_id}:")
        for label, num in counts_per_class.items():
            print(f"{label}:  {num}")

        return ImageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, debug_image
        )
