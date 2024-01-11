from tasks.point_extraction.entities import MapTile, MapTiles, MapPointLabel
from tasks.point_extraction.pytorch.mobilenet_rcnn import MobileNetRCNN
from tasks.point_extraction.pytorch.utils import PointInferenceDataset

from tasks.common.s3_data_cache import S3DataCache
from tasks.common.task import Task, TaskInput, TaskResult

import os
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


class MobileNetPointDetector(Task):
    """
    Model for detecting points in images. Predicts the location of points using a Pytorch model.
    Predicts directionality features of each point, if necessary, using a separate model.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _VERSION = 1
    _PYTORCH_MODEL = MobileNetRCNN

    LABEL_MAPPING = {
        "mine_pt": 1,
        "stike_and_dip": 2,
        "strikedip": 3,
        "strike_dip": 4,
        "inclined_bedding": 5,
        "overturned_bedding": 6,
        "gravel_pit_pt": 7,
    }

    def __init__(self, task_id: str, path: str) -> None:
        self.model = self._PYTORCH_MODEL.from_pretrained(path)
        self.model.eval()
        self.model.to(self.device)
        super().__init__(task_id)

    @staticmethod
    def dataloader_factory(images: List[MapTile]) -> DataLoader:
        dataset = PointInferenceDataset(tiles=images)
        return DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=default_collate,
        )

    @staticmethod
    def model_factory(model_name: str):
        raise NotImplementedError

    def reformat_output(self, output: Dict) -> List[MapPointLabel]:
        """
        Reformats Pytorch model output to match the MapPointLabel schema.
        """

        formatted_data: List[MapPointLabel] = []

        id_to_name = {v: k for k, v in self.LABEL_MAPPING.items()}

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            formatted_data.append(
                MapPointLabel(
                    classifier_name=type(self).__name__,
                    classifier_version=self._VERSION,
                    class_id=label,
                    class_name=id_to_name[label],
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    score=float(score),
                )
            )
        return formatted_data

    def to(self, device: str) -> None:
        """
        Move underlying Pytorch model to specified device.
        """
        self.model.to(device)

    @classmethod
    def load(cls, task_id: str, path: str):
        return cls(task_id, path)

    @property
    def version(self):
        return self._VERSION

    @property
    def input_type(self):
        return List[MapTile]

    @property
    def output_type(self):
        return List[MapTile]

    def run(
        self,
        input: TaskInput,
    ) -> TaskResult:
        """
        Prediction utility for inference and evaluation.
        """
        map_tiles = MapTiles.model_validate(input.data["map_tiles"])
        dataloader = self.dataloader_factory(images=map_tiles.tiles)

        predictions: List[MapTile] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Running {type(self).__name__}"):
                images, metadata = batch
                outputs = self.model(images)
                for idx, image in enumerate(images):
                    predictions.append(
                        MapTile(
                            x_offset=int(metadata["x_offset"][idx].item()),
                            y_offset=int(metadata["y_offset"][idx].item()),
                            width=int(metadata["width"][idx].item()),
                            height=int(metadata["height"][idx].item()),
                            image=image,
                            map_path=metadata["map_path"][idx],
                            predictions=self.reformat_output(outputs[idx]),
                        )
                    )
        result_map_tiles = MapTiles(raster_id=map_tiles.raster_id, tiles=predictions)
        return TaskResult(
            task_id=self._task_id,
            output={"map_tiles": result_map_tiles.model_dump()},
        )


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
        bsz: int = 15,
        device: str = "auto",
    ):
        local_data_path = self._prep_model_data(model_data_path, cache_path)
        self.model = YOLO(local_data_path)
        self.bsz = bsz
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
            local_model_data_path = Path(model_data_path)

        # check that we have all the files we need
        if not local_model_data_path or not local_model_data_path.is_file():
            raise ValueError(f"Model weights file not found at {model_data_path}")

        return local_model_data_path

    def process_output(self, predictions: Results) -> List[MapPointLabel]:
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
        map_tiles = MapTiles.model_validate(input.data["map_tiles"])

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        output: List[MapTile] = []
        for i in tqdm(range(0, len(map_tiles.tiles), self.bsz)):
            print(f"Processing batch {i} to {i + self.bsz}")
            batch = map_tiles.tiles[i : i + self.bsz]
            images = [tile.image for tile in batch]
            batch_preds = self.model.predict(images, imgsz=768, device=self.device)
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
