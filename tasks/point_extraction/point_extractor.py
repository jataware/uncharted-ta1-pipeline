from tasks.point_extraction.entities import MapTile, MapTiles, MapPointLabel
from tasks.common.s3_data_cache import S3DataCache
from tasks.common.task import Task, TaskInput, TaskResult
from enum import Enum
import logging
import os
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


logger = logging.getLogger(__name__)


class POINT_CLASS(str, Enum):
    STRIKE_AND_DIP = "strike_and_dip"  # aka inclined bedding
    HORIZONTAL_BEDDING = "horizontal_bedding"
    OVERTURNED_BEDDING = "overturned_bedding"
    VERTICAL_BEDDING = "vertical_bedding"
    INCLINED_FOLIATION = "inclined_foliation"
    VERTICAL_FOLIATION = "vertical_foliation"
    VERTICAL_JOINT = "vertical_joint"
    SINK_HOLE = "sink_hole"
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

    _VERSION = 1

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

        super().__init__(task_id)

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

    def process_output(self, predictions: Results) -> List[MapPointLabel]:
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
                pt_labels.append(
                    MapPointLabel(
                        classifier_name="unchartNet_point_extractor",
                        classifier_version=self._VERSION,
                        class_id=int(class_id),
                        class_name=self.model.names[int(class_id)],
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        score=score,
                    )
                )
        return pt_labels

    def run(self, input: TaskInput) -> TaskResult:
        """
        run YOLO model inference for point symbol detection
        """
        map_tiles = MapTiles.model_validate(input.data["map_tiles"])

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        output: List[MapTile] = []
        # run batch model inference...
        for i in tqdm(range(0, len(map_tiles.tiles), self.bsz)):
            logger.info(f"Processing batch {i} to {i + self.bsz}")
            batch = map_tiles.tiles[i : i + self.bsz]
            images = [tile.image for tile in batch]
            # note: ideally tile sizes used should be the same size as used during model training
            # tiles can be resized during inference pre-processing, if needed using 'imgsz' param
            # (e.g., predict(... imgsz=[1024,1024]))
            batch_preds = self.model.predict(images, device=self.device)
            for tile, preds in zip(batch, batch_preds):
                tile.predictions = self.process_output(preds)
                output.append(tile)
        result_map_tiles = MapTiles(raster_id=map_tiles.raster_id, tiles=output)
        return TaskResult(
            task_id=self._task_id, output={"map_tiles": result_map_tiles.model_dump()}
        )

    @property
    def version(self):
        return self._VERSION

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return List[MapTile]
