from tasks.point_extraction.entities import (
    MapTile,
    MapTiles,
    MapPointLabel,
    LegendPointItem,
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
)
from tasks.point_extraction.point_extractor_utils import find_legend_label_matches

from tasks.common.s3_data_cache import S3DataCache
from tasks.common.task import Task, TaskInput, TaskResult
from enum import Enum
import hashlib
import logging
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

# YOLO inference hyperparameters, https://docs.ultralytics.com/modes/predict/#inference-arguments
# CONF_THRES = 0.20  # (0.25) minimum confidence threshold for detections
# IOU_THRES = 0.7  # IoU threshold for NMS

logger = logging.getLogger(__name__)


class POINT_CLASS(str, Enum):
    STRIKE_AND_DIP = "strike_and_dip"  # aka inclined bedding
    HORIZONTAL_BEDDING = "horizontal_bedding"
    OVERTURNED_BEDDING = "overturned_bedding"
    VERTICAL_BEDDING = "vertical_bedding"
    INCLINED_FOLIATION = "inclined_foliation"  # line with solid triangle
    INCLINED_FOLIATION_IGNEOUS = "inclined_foliation_igneous"  # with hollow triangle
    VERTICAL_FOLIATION = "vertical_foliation"
    VERTICAL_JOINT = "vertical_joint"
    SINK_HOLE = "sink_hole"
    LINEATION = "lineation"
    GRAVEL_BORROW_PIT = "gravel_borrow_pit"
    MINE_SHAFT = "mine_shaft"
    PROSPECT = "prospect"
    MINE_TUNNEL = "mine_tunnel"  # aka adit or "4_pt"
    MINE_QUARRY = "mine_quarry"

    def __str__(self):
        return self.value


class YOLOPointDetector(Task):
    """
    Wrapper for Ultralytics YOLOv8 model inference.
    """

    def __init__(
        self,
        task_id: str,
        model_data_path: str,
        cache_path: str,
        batch_size: int = 20,
        device: str = "auto",
    ):
        local_data_path = self._prep_model_data(model_data_path, cache_path)
        self.model = YOLO(local_data_path)
        self.bsz = batch_size
        self.device = device
        self._model_id = self._get_model_id(self.model)

        super().__init__(task_id, cache_path)

    def _prep_model_data(self, model_data_path: str, data_cache_path: str) -> Path:
        """
        prepare local data cache and download model weights, if needed

        Args:
            model_data_path (str): The path to the model weights file
            data_cache_path (str): The path to the local data cache

        Returns:
            Path: The path to the local copy of the model weights file
        """

        local_model_data_path = None

        # check if path is a URL
        if model_data_path.startswith("s3://") or model_data_path.startswith("http"):
            # need to specify data cache path when fetching from S3
            if data_cache_path == "":
                raise ValueError(
                    "'data_cache_path' must be specified when fetching model data from S3"
                )

            s3_host = ""
            s3_path = ""
            s3_bucket = ""

            res = urlparse(model_data_path)
            s3_host = res.scheme + "://" + res.netloc
            s3_path = res.path.lstrip("/")
            s3_bucket = s3_path.split("/")[0]
            s3_path = s3_path.lstrip(s3_bucket)
            s3_path = s3_path.lstrip("/")

            # create local data cache, if doesn't exist, and connect to S3
            s3_data_cache = S3DataCache(
                data_cache_path,
                s3_host,
                s3_bucket,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "<UNSET>"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "<UNSET>"),
            )

            # get teh model weights file either from the locally cached copy or from S3
            local_model_data_path = Path(
                s3_data_cache.fetch_file_from_s3(s3_path, overwrite=False)
            )
        else:
            # load the model weights file from the local filesystem
            logger.info(f"Loading model weights from local path: {model_data_path}")
            local_model_data_path = Path(model_data_path)

        # check that we have all the files we need
        if not local_model_data_path or not local_model_data_path.is_file():
            raise ValueError(f"Model weights file not found at {model_data_path}")

        return local_model_data_path

    def process_output(
        self, predictions: Results, point_legend_mapping: Dict[str, LegendPointItem]
    ) -> List[MapPointLabel]:
        """
        Convert point detection inference results from YOLO model format
        to a list of MapPointLabel objects
        """
        pt_labels = []
        for pred in predictions:
            assert pred.boxes is not None
            assert isinstance(pred.boxes.data, torch.Tensor)
            assert isinstance(self.model.names, Dict)

            if len(pred.boxes.data) == 0:
                continue
            for box in pred.boxes.data.detach().cpu().tolist():
                x1, y1, x2, y2, score, class_id = box

                # map YOLO class name to legend item name, if available
                class_name = self.model.names[int(class_id)]
                legend_name = class_name
                legend_bbox = []
                if class_name in point_legend_mapping:
                    legend_name = point_legend_mapping[class_name].name
                    legend_bbox = point_legend_mapping[class_name].legend_bbox

                pt_labels.append(
                    MapPointLabel(
                        classifier_name="unchartNet_point_extractor",
                        classifier_version=self._model_id,
                        class_id=int(class_id),
                        class_name=self.model.names[int(class_id)],
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        score=score,
                        legend_name=legend_name,
                        legend_bbox=legend_bbox,
                    )
                )
        return pt_labels

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run YOLO model inference for point symbol detection
        """
        map_tiles = MapTiles.model_validate(task_input.data["map_tiles"])

        point_legend_mapping: Dict[str, LegendPointItem] = {}
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
            # find mappings between legend item labels and YOLO model class names
            point_legend_mapping = find_legend_label_matches(
                legend_pt_items, task_input.raster_id
            )

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        doc_key = f"{task_input.raster_id}_points-{self._model_id}"
        # check cache and re-use existing file if present
        json_data = self.fetch_cached_result(doc_key)
        if json_data and map_tiles.join_with_cached_predictions(
            MapTiles(**json_data), point_legend_mapping
        ):
            # cached point predictions loaded successfully
            logger.info(
                f"Using cached point extractions for raster: {task_input.raster_id}"
            )
            return TaskResult(
                task_id=self._task_id,
                output={"map_tiles": map_tiles},
            )

        output: List[MapTile] = []
        # run batch model inference...
        for i in tqdm(range(0, len(map_tiles.tiles), self.bsz)):
            logger.info(f"Processing batch {i} to {i + self.bsz}")
            batch = map_tiles.tiles[i : i + self.bsz]
            images = [tile.image for tile in batch]
            # note: ideally tile sizes used should be the same size as used during model training
            # tiles can be resized during inference pre-processing, if needed using 'imgsz' param
            # (e.g., predict(... imgsz=[1024,1024]))
            batch_preds = self.model.predict(
                images,
                device=self.device,
                # conf=CONF_THRES,
                # iou=IOU_THRES,
            )
            for tile, preds in zip(batch, batch_preds):
                tile.predictions = self.process_output(preds, point_legend_mapping)
                output.append(tile)
        result_map_tiles = MapTiles(raster_id=map_tiles.raster_id, tiles=output)

        # write to cache
        self.write_result_to_cache(
            result_map_tiles.format_for_caching().model_dump(), doc_key
        )

        return TaskResult(
            task_id=self._task_id, output={"map_tiles": result_map_tiles.model_dump()}
        )

    def _get_model_id(self, model: YOLO) -> str:
        """
        Create a unique string ID for this model,
        based on MD5 hash of the model's state-dict
        """
        state_dict_str = str(model.state_dict())
        hash_result = hashlib.md5(bytes(state_dict_str, encoding="utf-8"))
        return hash_result.hexdigest()

    @property
    def version(self):
        return self._model_id

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return List[MapTile]
