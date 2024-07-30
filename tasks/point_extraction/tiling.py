from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import logging
import cv2

from common.task import Task, TaskInput, TaskResult

from tasks.point_extraction.entities import (
    ImageTile,
    ImageTiles,
    PointLabels,
    PointLabel,
)
from tasks.segmentation.entities import (
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
    SEGMENT_MAP_CLASS,
    SEGMENT_POINT_LEGEND_CLASS,
)
from tasks.segmentation.segmenter_utils import get_segment_bounds, segments_to_mask
from tasks.point_extraction.entities import (
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
    MAP_TILES_OUTPUT_KEY,
    LEGEND_TILES_OUTPUT_KEY,
    MAP_PT_LABELS_OUTPUT_KEY,
    LEGEND_PT_LABELS_OUTPUT_KEY,
)
from tasks.point_extraction.legend_item_utils import legend_items_use_ontology

TILE_OVERLAP_DEFAULT = (  # default tliing overlap = point bbox + 10%
    int(1.1 * 90),
    int(1.1 * 90),
)
DEDUP_DECIMATION_FACTOR = 0.5  # de-dup predictions (in tile overlap regions) if bbox centers are within 2 pixels

logger = logging.getLogger(__name__)


class Tiler(Task):
    """
    Decomposes a full image into smaller tiles

    NOTE: for point extractor inference, for best results it is recommended
    to use the same size tiles that were used during model training
    e.g., 1024x1024
    """

    # TODO: handle case where image already has labels attached to it.
    def __init__(
        self,
        task_id="",
        tile_size: tuple = (1024, 1024),
        overlap: tuple = TILE_OVERLAP_DEFAULT,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        super().__init__(task_id)

    def run(
        self,
        task_input: TaskInput,
    ) -> TaskResult:
        image_array = np.array(task_input.image)
        x_min = 0
        y_min = 0
        y_max, x_max, _ = image_array.shape

        # ---- use image segmentation to restrict point extraction to map area only
        poly_legend = []
        if SEGMENTATION_OUTPUT_KEY in task_input.data:
            segmentation = MapSegmentation.model_validate(
                task_input.data[SEGMENTATION_OUTPUT_KEY]
            )
            # get a binary mask of the regions-of-interest and apply to the input image before tiling
            binary_mask = segments_to_mask(
                segmentation,
                (x_max, y_max),
                roi_classes=[SEGMENT_MAP_CLASS, SEGMENT_POINT_LEGEND_CLASS],
            )
            if binary_mask.size != 0:
                # apply binary mask to input image prior to tiling
                image_array = cv2.bitwise_and(
                    image_array, image_array, mask=binary_mask
                )

            poly_map = get_segment_bounds(segmentation, SEGMENT_MAP_CLASS)
            poly_legend = get_segment_bounds(segmentation, SEGMENT_POINT_LEGEND_CLASS)
            if len(poly_map) > 0:
                # restrict tiling to use *only* the bounding rectangle of map area
                poly_map = poly_map[0]  # use 1st (highest ranked) map segment
                (x_min, y_min, x_max, y_max) = [int(b) for b in poly_map.bounds]
        roi_bounds = (x_min, y_min, x_max, y_max)

        # ---- create tiles for map area
        logger.info("Creating map area tiles")
        map_tiles = self._create_tiles(
            task_input.raster_id, image_array, [roi_bounds], SEGMENT_MAP_CLASS
        )

        # --- load legend item annotations, if available
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
            if poly_legend and not legend_items_use_ontology(legend_pt_items):
                # legend annotations don't all use the expected ontology
                roi_bounds = []
                for p_leg in poly_legend:
                    roi_bounds.append([int(b) for b in p_leg.bounds])
                logger.info("Also creating legend area tiles")
                legend_tiles = self._create_tiles(
                    task_input.raster_id,
                    image_array,
                    roi_bounds,
                    SEGMENT_POINT_LEGEND_CLASS,
                )

                # prepare task result with both map and legend tiles
                return TaskResult(
                    task_id=self._task_id,
                    output={
                        MAP_TILES_OUTPUT_KEY: map_tiles.model_dump(),
                        LEGEND_TILES_OUTPUT_KEY: legend_tiles.model_dump(),
                    },
                )

        # prepare task result with only map tiles
        return TaskResult(
            task_id=self._task_id, output={MAP_TILES_OUTPUT_KEY: map_tiles.model_dump()}
        )

    def _create_tiles(
        self,
        raster_id: str,
        image_array: np.ndarray,
        roi_bounds: List[Tuple],
        roi_label: str,
    ) -> ImageTiles:
        """
        create tiles for an image regions-of-interest
        """
        step_x = int(self.tile_size[0] - self.overlap[0])
        step_y = int(self.tile_size[1] - self.overlap[1])
        tiles: List[ImageTile] = []

        for bounds in roi_bounds:
            (x_min, y_min, x_max, y_max) = bounds

            for y in range(y_min, y_max, step_y):
                for x in range(x_min, x_max, step_x):
                    width = min(self.tile_size[0], x_max - x)
                    height = min(self.tile_size[1], y_max - y)

                    tile_array = image_array[y : y + height, x : x + width]  # type: ignore

                    if (
                        tile_array.shape[0] < self.tile_size[1]
                        or tile_array.shape[1] < self.tile_size[0]
                    ):
                        padded_tile = np.zeros(
                            (self.tile_size[1], self.tile_size[0], 3),
                            dtype=tile_array.dtype,
                        )

                        padded_tile[:height, :width] = tile_array
                        tile_array = padded_tile

                    maptile = ImageTile(
                        x_offset=x,
                        y_offset=y,
                        width=self.tile_size[0],
                        height=self.tile_size[1],
                        image=Image.fromarray(tile_array),
                        image_path="",
                    )
                    tiles.append(maptile)
        # get global bounds, if multiple segments for this roi
        bounds_global = []
        if len(roi_bounds) > 0:
            bounds_global = list(roi_bounds[0])
            for bounds in roi_bounds:
                bounds_global[0] = min(bounds_global[0], bounds[0])
                bounds_global[1] = min(bounds_global[1], bounds[1])
                bounds_global[2] = max(bounds_global[2], bounds[2])
                bounds_global[3] = max(bounds_global[3], bounds[3])

        image_tiles = ImageTiles(
            raster_id=raster_id,
            tiles=tiles,
            roi_bounds=tuple(bounds_global),
            roi_label=roi_label,
        )

        return image_tiles

    @property
    def input_type(self):
        return PointLabels

    @property
    def output_type(self):
        return List[ImageTile]


class Untiler(Task):
    def __init__(self, task_id="", overlap: tuple = TILE_OVERLAP_DEFAULT):
        # NOTE: Untiler recommended to use the same tile overlap as corresponding Tiler class instance
        self.overlap = overlap
        super().__init__(task_id)

    """
    Used to reconstruct the original image from the tiles and map back the bounding boxes and labels.
    Note that new images aren't actually constructed here, we are just mapping predictions from tiles onto the original map.
    """

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Reconstructs the original image from the tiles and maps back the bounding boxes and labels.
        tile_predictions: List of PointLabel objects. Generated by the model. TILES MUST BE FROM ONLY ONE MAP.
        returns: List of PointLabel objects. These can be mapped directly onto the original map.
        """

        # run untiling on map tiles
        logger.info("Untiling map tiles...")
        map_tiles = ImageTiles.model_validate(task_input.data[MAP_TILES_OUTPUT_KEY])
        map_point_labels = self._merge_tiles(map_tiles, task_input.raster_id)

        if LEGEND_TILES_OUTPUT_KEY in task_input.data:
            # --- also run untiling on legend area tiles, if available
            logger.info("Also untiling legend area tiles...")
            legend_tiles = ImageTiles.model_validate(
                task_input.data[LEGEND_TILES_OUTPUT_KEY]
            )
            legend_point_labels = self._merge_tiles(legend_tiles, task_input.raster_id)

            # store untiling results for both map and legend areas
            return TaskResult(
                task_id=self._task_id,
                output={
                    MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump(),
                    LEGEND_PT_LABELS_OUTPUT_KEY: legend_point_labels.model_dump(),
                },
            )

        # store untiling results for map area
        return TaskResult(
            task_id=self._task_id,
            output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
        )

    def _merge_tiles(self, image_tiles: ImageTiles, raster_id: str) -> PointLabels:
        """
        Merge tile extractions by converting predictions results to global image pixel coordinates
        """

        assert all(
            i.predictions is not None for i in image_tiles.tiles
        ), "Tiles must have predictions attached to them."
        all_predictions = []
        overlap_predictions = {}
        num_dedup = 0
        map_path = image_tiles.tiles[0].image_path
        for tile in image_tiles.tiles:

            x_offset = tile.x_offset  # xmin of tile, absolute value in original map
            y_offset = tile.y_offset  # ymin of tile, absolute value in original map

            for pred in tqdm(
                tile.predictions,
                desc="Reconstructing original image with predictions on tiles",
            ):

                x1 = pred.x1
                x2 = pred.x2
                y1 = pred.y1
                y2 = pred.y2
                score = pred.score
                label_name = pred.class_name

                # filter noisy predictions due to tile overlap
                pred_redundant = False
                pred_in_overlap = False
                if self.overlap[0] > 0 or self.overlap[1] > 0:
                    pred_redundant, pred_in_overlap = self._is_prediction_redundant(
                        pred,
                        image_tiles.roi_bounds,
                        (tile.x_offset, tile.y_offset),
                        (tile.width, tile.height),
                    )

                if pred_redundant:
                    continue

                global_prediction = PointLabel(
                    model_name=pred.model_name,
                    model_version=pred.model_version,
                    class_id=pred.class_id,
                    class_name=label_name,
                    # Add offset of tile to project onto original map...
                    x1=x1 + x_offset,
                    y1=y1 + y_offset,
                    x2=x2 + x_offset,
                    y2=y2 + y_offset,
                    score=score,
                    direction=pred.direction,
                    dip=pred.dip,
                )

                if pred_in_overlap:
                    # store predictions in overlapped tile regions in a dict, to de-duplicate as needed
                    xc_idx = ((x1 + x2) / 2) + x_offset
                    yc_idx = ((y1 + y2) / 2) + y_offset
                    pred_key = (
                        label_name,
                        int(xc_idx * DEDUP_DECIMATION_FACTOR),
                        int(yc_idx * DEDUP_DECIMATION_FACTOR),
                    )
                    if pred_key in overlap_predictions:
                        # duplicate prediction will be overwritten here
                        num_dedup += 1
                    overlap_predictions[pred_key] = global_prediction
                else:
                    all_predictions.append(global_prediction)

        # merge de-dup'd predictions into final list for whole map
        all_predictions.extend(list(overlap_predictions.values()))

        logger.info(
            f"Total point predictions after re-constructing image tiles: {len(all_predictions)}, with {num_dedup} discarded as duplicates"
        )

        return PointLabels(path=map_path, raster_id=raster_id, labels=all_predictions)

    def _is_prediction_redundant(
        self,
        pred: PointLabel,
        roi_bbox,
        tile_offset: tuple,
        tile_wh: tuple,
        shape_thres=2,
        conf_thres=0.5,
    ) -> tuple:
        """
        Check if a point symbol prediction is redundant
        (based on heuristic of bbox shape and location at tile edge)
        """
        x1 = pred.x1
        x2 = pred.x2
        y1 = pred.y1
        y2 = pred.y2
        (roi_xmin, roi_ymin, roi_xmax, roi_ymax) = roi_bbox
        tile_w, tile_h = tile_wh
        x_offset, y_offset = tile_offset

        pred_redundant = False
        pred_in_overlap = False
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        # check if the prediction is in a region where multiple tiles overlap
        if (
            xc < self.overlap[0]
            or xc > tile_wh[0] - self.overlap[0]
            or yc < self.overlap[1]
            or yc > tile_wh[1] - self.overlap[1]
        ):
            pred_in_overlap = True

        # TODO - instead of checking at tile edge could check if bbox edge is in overlap region
        if (abs((x2 - x1) - (y2 - y1)) > shape_thres) and (
            x1 <= 1 or y1 <= 1 or x2 >= tile_w - 1 or y2 >= tile_h - 1
        ):
            # pred bbox is at a tile edge and NOT square,
            # check if bbox edges correspond to global image bounds
            if (
                x1 + x_offset > roi_xmin
                and x2 + x_offset < roi_xmax
                and y1 + y_offset > roi_ymin
                and y2 + y_offset < roi_ymax
            ):
                # non-square point bbox not at map edges, assume this is a noisy prediction (due to tile overlap) and skip
                pred_redundant = True
            elif pred.score < conf_thres:
                # non-square point bbox at map edges, and low confidence,
                # discard as a redundant or noisy prediction
                pred_redundant = True

        return (pred_redundant, pred_in_overlap)

    @property
    def input_type(self):
        return List[ImageTile]

    @property
    def output_type(self):
        return PointLabels
