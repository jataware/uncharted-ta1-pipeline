import logging
import cv2
from cv2.typing import MatLike
import numpy as np
import torch
import hashlib
from pathlib import Path
import os
from urllib.parse import urlparse
from typing import List, Tuple, Sequence

from tasks.common.io import Mode, get_file_source
from tasks.segmentation.ditod import add_vit_config
from tasks.segmentation.entities import (
    SegmentationResult,
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
)
from tasks.segmentation.segmenter_utils import rank_segments

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

        # check for conflicting paths - can't have model weights and cache both in S3
        # because S3 model weights are downloaded and stored on the local filesystem
        model_source = get_file_source(model_data_path)
        cache_source = get_file_source(model_data_cache_path)
        if (
            model_source == Mode.S3_URI or model_source == Mode.URL
        ) and cache_source == Mode.S3_URI:
            raise ValueError(
                "Model data path and cache path cannot both be S3 locations"
            )

        model_paths = self._prep_config_data(model_data_path, model_data_cache_path)

        self.config_file = str(model_paths.model_config_path)
        self.model_weights = str(model_paths.model_weights_path)
        self.class_labels = class_labels
        self.gpu = gpu
        self.confidence_thres = confidence_thres

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
        # note, path to model weights (e.g., model_final.pth), can be local file path or URL
        self.cfg.MODEL.WEIGHTS = self.model_weights
        logger.info(f"Using model weights at {self.model_weights}")

        # confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_thres

        # set device
        device = "cuda" if self.gpu == True and torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        logger.info(f"torch device: {device}")

        self._model_id = self._get_model_id()

        # load the segmentation model...
        logger.info(f"Loading segmentation model {self.model_name}")
        self.predictor = DefaultPredictor(self.cfg)

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

        # check cache and re-use existing file if present
        doc_key = f"{input.raster_id}_segmentation-{self._model_id}"
        cached_data = self.fetch_cached_result(doc_key)
        if cached_data:
            logger.info(
                f"Using cached segmentation results for raster: {input.raster_id}"
            )
            result = self._create_result(input)

            # load and post-process the cached segmentation result
            map_segmentation = MapSegmentation.model_validate(cached_data)
            rank_segments(map_segmentation, self.class_labels)

            result.add_output(
                SEGMENTATION_OUTPUT_KEY,
                map_segmentation.model_dump(),
            )
            return result

        # --- run inference
        predictions = self.predictor(np.array(input.image))["instances"]
        predictions = predictions.to("cpu")

        if (
            not predictions
            or not predictions.has("scores")
            or not predictions.has("pred_classes")
            or not predictions.has("pred_masks")
        ):
            logger.warning(
                "No valid segmentation predictions for this image!  Returning empty results"
            )
            map_segmentation = MapSegmentation(doc_id=input.raster_id, segments=[])
            # write empty results to cache
            self.write_result_to_cache(map_segmentation, doc_key)
            result = self._create_result(input)
            result.add_output(SEGMENTATION_OUTPUT_KEY, map_segmentation.model_dump())
            return result

        # convert prediction masks to polygons and prepare results
        scores = predictions.scores.tolist()
        classes = predictions.pred_classes.tolist()

        masks = np.asarray(predictions.pred_masks)

        # TODO -- could simplify the polygon segmentation result, if desired (fewer keypoints, etc.)
        # https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html#shapely.Polygon.simplify
        seg_results: List[SegmentationResult] = []
        for i, mask in enumerate(masks):
            contours, hierarchies = self._mask_to_contours(mask)
            # hierarchy is opencv contours format,
            # for each contour the hierarchy info is: [Next, Previous, First_Child, Parent]
            # (with -1 == null entry)
            # more info here: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
            for contour, hierarchy in zip(contours, hierarchies):
                if contour.size < 6:
                    # contour too small (ie just a line segment?); discard
                    continue
                # check if this contour is a "child" of another
                # (ie representing a hole in the segmentation mask)
                is_child = hierarchy[3] > -1
                if is_child:
                    # skip child contours
                    continue
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

        # rank the segments per class (most impt first)
        rank_segments(map_segmentation, self.class_labels)

        # write to cache
        self.write_result_to_cache(map_segmentation, doc_key)

        result = self._create_result(input)
        result.add_output(SEGMENTATION_OUTPUT_KEY, map_segmentation.model_dump())
        return result

    def _mask_to_contours(
        self, mask: np.ndarray
    ) -> Tuple[Sequence[MatLike], Sequence[MatLike]]:
        """
        Converts segmentation mask to polygon contours
        Adapted from Detectron2 GenericMask 'mask_to_polygons' code

        Returns a tuple of (contours, contour hierarchy info)
        """

        mask = np.ascontiguousarray(
            mask
        )  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], []
        # convert hierarchy info to a list of 1d arrays
        hierarchy = list(hierarchy.reshape(-1, 4))
        contours = res[-2]
        return (contours, hierarchy)

    def _get_model_id(self) -> str:
        """
        Create a unique string ID for this model,
        based on MD5 hash of the model's state-dict
        """
        attributes = "_".join(
            [
                "segmentation",
                self.model_weights,
                self.config_file,
                str(self.confidence_thres),
                "_".join(self.class_labels),
            ]
        )
        return hashlib.sha256(attributes.encode()).hexdigest()

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
