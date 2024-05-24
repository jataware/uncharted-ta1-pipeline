import logging
import cv2
from cv2.typing import MatLike
import numpy as np
from sympy import LM
import torch
import hashlib
from pathlib import Path
import os
from urllib.parse import urlparse
from typing import List, Optional, Tuple, Sequence

from tasks.segmentation.ditod import add_vit_config
from tasks.segmentation.entities import (
    SegmentationResult,
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
)
from detectron2.config import get_cfg
from detectron2.layers import mask_ops
from detectron2.engine import DefaultPredictor
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.common.s3_data_cache import S3DataCache

# Internal Detectron GPU mem threshold
# TODO: could try to tune dynamically, based on available resources
mask_ops.GPU_MEM_LIMIT = 2 * (1024**3)  # 2GB (default=1GB)

CONFIDENCE_THRES_DEFAULT = 0.25  # default confidence threshold (model will discard any regions with confidence < threshold)

THING_CLASSES_DEFAULT = [
    "cross_section",
    "legend_points_lines",
    "legend_polygons",
    "map",
]  # default mapping of segmentation classes -> labels

# model support files
MODEL_FILENAME = "model_final.pth"
LM_CONFIG_FILENAME = "config.yaml"
DET_CONFIG_FILENAME = "config.json"

logger = logging.getLogger(__name__)


class ModelPaths:
    def __init__(self, model_weights_path: Path, model_config_path: Path):
        self.model_weights_path = model_weights_path
        self.model_config_path = model_config_path


class DetectronSegmenter(Task):
    """
    Class to handle inference for Legend and Map image segmentation
    using a Detectron2-based model, such as LayoutLMv3
    """

    def __init__(
        self,
        task_id: str,
        model_data_path: str,
        model_data_cache_path: str,
        class_labels: list = THING_CLASSES_DEFAULT,
        confidence_thres: float = CONFIDENCE_THRES_DEFAULT,
        gpu: bool = True,
    ):
        super().__init__(task_id, model_data_cache_path)

        model_paths = self._prep_config_data(model_data_path, model_data_cache_path)

        self.config_file = str(model_paths.model_config_path)
        self.model_weights = str(model_paths.model_weights_path)
        self.class_labels = class_labels
        self.gpu = gpu

        # instantiate config
        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(self.config_file)  # config yml file
        self.model_name = self.cfg.MODEL.VIT.get("NAME", "")

        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 3 and len(self.class_labels) > 3:
            # backwards compatibility for older 3-class segmentation model
            # (ie without map cross-section segmentation)
            self.class_labels = [
                "legend_points_lines",
                "legend_polygons",
                "map",
            ]

        # add model weights URL to config
        self.cfg.MODEL.WEIGHTS = (
            self.model_weights
        )  # path to model weights (e.g., model_final.pth), can be local file path or URL
        logger.info(f"Using model weights at {self.model_weights}")
        # TODO use a local cache to check for existing model weights (instead of re-downloading each time?)

        # confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thres

        # set device
        device = "cuda" if self.gpu == True and torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        logger.info(f"torch device: {device}")

        # load the segmentation model...
        logger.info(f"Loading segmentation model {self.model_name}")
        self.predictor = DefaultPredictor(self.cfg)
        self._model_id = self._get_model_id(self.predictor.model)
        logger.info(f"Model ID: {self._model_id }")

    def run(self, input: TaskInput) -> TaskResult:
        """
        Run legend and map segmentation inference on a single input image
        Note: model is lazily loaded into memory the first time this func is called

        Args:
            input: TaskInput object with image to process

        Returns:
            List of SegmentationResult objects
        """

        # using Detectron2 DefaultPredictor class for model inference
        # TODO -- switch to using detectron2 model API directly for inference on batches of images
        # https://detectron2.readthedocs.io/en/latest/tutorials/models.html

        doc_key = f"{input.raster_id}_segmentation-{self._model_id}"

        # check cache and re-use existing file if present
        json_data = self.fetch_cached_result(doc_key)
        if json_data:
            logger.info(
                f"Using cached segmentation results for raster: {input.raster_id}"
            )
            result = self._create_result(input)
            result.add_output(
                SEGMENTATION_OUTPUT_KEY,
                MapSegmentation(**json_data).model_dump(),
            )
            return result

        # --- run inference
        predictions = self.predictor(np.array(input.image))["instances"]
        predictions = predictions.to("cpu")

        if not predictions:
            logger.warn("No segmentation predictions for this image!")
            return self._create_result(input)

        if (
            not predictions.has("scores")
            or not predictions.has("pred_classes")
            or not predictions.has("pred_masks")
        ):
            logger.warn(
                "Segmentation predictions are missing data or format is unexpected! Returning empty results"
            )
            return self._create_result(input)

        # convert prediction masks to polygons and prepare results
        scores = predictions.scores.tolist()
        classes = predictions.pred_classes.tolist()

        masks = np.asarray(predictions.pred_masks)

        # TODO -- could simplify the polygon segmentation result, if desired (fewer keypoints, etc.)
        # https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html#shapely.Polygon.simplify

        seg_results: List[SegmentationResult] = []
        for i, mask in enumerate(masks):
            contours, _ = self._mask_to_contours(mask)

            for contour in contours:
                if contour.size >= 6:
                    poly = contour.reshape(-1, 2)
                    seg_result = SegmentationResult(
                        poly_bounds=poly.tolist(),
                        bbox=list(cv2.boundingRect(contour)),
                        area=cv2.contourArea(contour),
                        confidence=scores[i],
                        class_label=self.class_labels[classes[i]],
                        id_model=self._model_id,
                    )
                    seg_results.append(seg_result)
        map_segmentation = MapSegmentation(doc_id=input.raster_id, segments=seg_results)
        json_data = map_segmentation.model_dump()

        # write to cache
        self.write_result_to_cache(json_data, doc_key)

        result = self._create_result(input)
        result.add_output(SEGMENTATION_OUTPUT_KEY, json_data)
        return result

    def run_inference_batch(self):
        """
        Run legend and map segmentation inference on a batch of images
        """
        # TODO add batch processing support (see comment above)
        raise NotImplementedError

    def _mask_to_contours(self, mask: np.ndarray) -> Tuple[Sequence[MatLike], bool]:
        """
        Converts segmentation mask to polygon contours
        Adapted from Detectron2 GenericMask code
        """

        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        reshaped: np.ndarray = hierarchy.reshape(-1, 4)
        has_holes = (reshaped[:, 3] >= 0).sum() > 0
        return (res[-2], has_holes)

    def _get_model_id(self, model) -> str:
        """
        Create a unique string ID for this model,
        based on MD5 hash of the model's state-dict
        """

        state_dict_str = str(model.state_dict())
        hash_result = hashlib.md5(bytes(state_dict_str, encoding="utf-8"))
        return hash_result.hexdigest()

    def _prep_config_data(
        self, model_data_path: str, data_cache_path: str
    ) -> ModelPaths:
        """
        prepare local data cache and download model weights, if needed

        Args:
            model_data_path (str): The path to the folder containing the model weights and config files
            data_cache_path (str): The path to the local data cache.

        Returns:
            ModelPaths: The paths to the model weights and config files.
        """

        local_model_data_path = None
        local_lm_config_path = None
        local_det_config_path = None

        # check if path is a URL

        if model_data_path.startswith("s3://") or model_data_path.startswith("http"):
            # need to specify data cache path when fetching from S3
            if not data_cache_path:
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

            # create local data cache, if doesn't exist
            s3_data_cache = S3DataCache(
                data_cache_path,
                s3_host,
                s3_bucket,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "<UNSET>"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "<UNSET>"),
            )

            # check for model weights and config files in the folder
            model_key = os.path.join(s3_path, MODEL_FILENAME)
            local_model_data_path = Path(
                s3_data_cache.fetch_file_from_s3(model_key, overwrite=False)
            )

            lm_config_key = os.path.join(s3_path, LM_CONFIG_FILENAME)
            local_lm_config_path = Path(
                s3_data_cache.fetch_file_from_s3(lm_config_key, overwrite=False)
            )

            det_config_key = os.path.join(s3_path, DET_CONFIG_FILENAME)
            local_det_config_path = Path(
                s3_data_cache.fetch_file_from_s3(det_config_key, overwrite=False)
            )
        else:
            # check for model weights and config files in the folder
            # iterate over files in folder
            for f in Path(model_data_path).iterdir():
                if f.is_file():
                    if f.name.endswith(MODEL_FILENAME):
                        local_model_data_path = f
                    elif f.name.endswith(LM_CONFIG_FILENAME):
                        local_lm_config_path = f
                    elif f.name.endswith(DET_CONFIG_FILENAME):
                        local_det_config_path = f

        # check that we have all the files we need
        if not local_model_data_path or not local_model_data_path.is_file():
            raise ValueError(f"Model weights file not found at {model_data_path}")

        if not local_det_config_path or not local_det_config_path.is_file():
            raise ValueError(f"Detectron config file not found at {model_data_path}")

        if not local_lm_config_path or not local_lm_config_path.is_file():
            raise ValueError(f"LayoutLM config file not found at {model_data_path}")

        return ModelPaths(local_model_data_path, local_lm_config_path)
