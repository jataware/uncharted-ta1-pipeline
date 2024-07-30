from tasks.point_extraction.entities import (
    ImageTile,
    ImageTiles,
    PointLabel,
    MAP_TILES_OUTPUT_KEY,
    LEGEND_TILES_OUTPUT_KEY,
)

from tasks.common.s3_data_cache import S3DataCache
from tasks.common.task import Task, TaskInput, TaskResult
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
CONF_THRES = 0.20  # (0.25) minimum confidence threshold for detections
IOU_THRES = 0.7  # IoU threshold for NMS

MODEL_NAME = "uncharted_ml_point_extractor"

logger = logging.getLogger(__name__)


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

    def process_output(self, predictions: Results) -> List[PointLabel]:
        """
        Convert point detection inference results from YOLO model format
        to a list of PointLabel objects
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
                class_name = self.model.names[int(class_id)]

                pt_labels.append(
                    PointLabel(
                        model_name=MODEL_NAME,
                        model_version=self._model_id,
                        class_id=int(class_id),
                        class_name=class_name,
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        score=score,
                    )
                )
        return pt_labels

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run YOLO model inference for point symbol detection
        """
        map_tiles = ImageTiles.model_validate(task_input.data[MAP_TILES_OUTPUT_KEY])

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        # --- run point extraction model on map area tiles
        logger.info(f"Running model inference on {len(map_tiles.tiles)} map tiles")
        self._process_tiles(map_tiles, task_input.raster_id, "map")

        if LEGEND_TILES_OUTPUT_KEY in task_input.data:
            # --- also run point extraction model on legend area tiles, if available
            legend_tiles = ImageTiles.model_validate(
                task_input.data[LEGEND_TILES_OUTPUT_KEY]
            )
            logger.info(
                f"Also running model inference on {len(legend_tiles.tiles)} legend tiles"
            )
            self._process_tiles(legend_tiles, task_input.raster_id, "legend")

            return TaskResult(
                task_id=self._task_id,
                output={
                    MAP_TILES_OUTPUT_KEY: map_tiles.model_dump(),
                    LEGEND_TILES_OUTPUT_KEY: legend_tiles.model_dump(),
                },
            )

        return TaskResult(
            task_id=self._task_id, output={MAP_TILES_OUTPUT_KEY: map_tiles.model_dump()}
        )

    def _process_tiles(
        self, image_tiles: ImageTiles, raster_id: str, tile_type: str = "map"
    ):
        """
        do batch inference on image tiles
        prediction results are appended in-place to the ImageTiles object
        """

        # get key for points' data cache
        doc_key = (
            f"{raster_id}_points-{self._model_id}"
            if tile_type == "map"
            else f"{raster_id}_points_{tile_type}-{self._model_id}"
        )
        # check cache and re-use existing file if present
        json_data = self.fetch_cached_result(doc_key)
        if json_data and image_tiles.join_with_cached_predictions(
            ImageTiles(**json_data)
        ):
            # cached point predictions loaded successfully
            logger.info(
                f"Using cached point extractions for raster {raster_id} and tile type {tile_type}"
            )
            return

        tiles_out: List[ImageTile] = []
        # run batch model inference...
        for i in tqdm(range(0, len(image_tiles.tiles), self.bsz)):
            logger.info(f"Processing batch {i} to {i + self.bsz}")
            batch = image_tiles.tiles[i : i + self.bsz]
            images = [tile.image for tile in batch]
            # note: ideally tile sizes used should be the same size as used during model training
            # tiles can be resized during inference pre-processing, if needed using 'imgsz' param
            # (e.g., predict(... imgsz=[1024,1024]))
            batch_preds = self.model.predict(
                images,
                device=self.device,
                conf=CONF_THRES,
                iou=IOU_THRES,
            )
            for tile, preds in zip(batch, batch_preds):
                tile.predictions = self.process_output(preds)
                tiles_out.append(tile)
        # save tile results with point extraction predictions
        image_tiles.tiles = tiles_out

        # write to cache
        self.write_result_to_cache(
            image_tiles.format_for_caching().model_dump(), doc_key
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
        return List[ImageTile]

    @property
    def output_type(self):
        return List[ImageTile]
