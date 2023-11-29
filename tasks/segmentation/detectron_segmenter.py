import logging
import cv2
from cv2.typing import MatLike
import numpy as np
import torch
import hashlib

from typing import List, Optional, Tuple, Sequence

from .ditod import add_vit_config
from .entities import SegmentationResult, MapSegmentation
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from common.task import Task, TaskInput, TaskResult

CONFIDENCE_THRES_DEFAULT = 0.25  # default confidence threshold (model will discard any regions with confidence < threshold)
THING_CLASSES_DEFAULT = [
    "legend_points_lines",
    "legend_polygons",
    "map",
]  # default mapping of segmentation classes -> labels

logger = logging.getLogger(__name__)


class DetectronSegmenter(Task):
    """
    Class to handle inference for Legend and Map image segmentation
    using a Detectron2-based model, such as LayoutLMv3
    """

    def __init__(
        self,
        task_id: str,
        config_file: str,
        model_weights: str,
        class_labels: list = THING_CLASSES_DEFAULT,
        confidence_thres: float = CONFIDENCE_THRES_DEFAULT,
    ):
        super().__init__(task_id)

        self.config_file = config_file
        self.model_weights = model_weights
        self.class_labels = class_labels
        self.predictor: Optional[DefaultPredictor] = None
        self.id_model: str = ""

        # instantiate config
        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(config_file)  # config yml file
        self.model_name = self.cfg.MODEL.VIT.get("NAME", "")

        # add model weights URL to config
        self.cfg.MODEL.WEIGHTS = model_weights  # path to model weights (e.g., model_final.pth), can be local file path or URL
        logger.info(f"Using model weights at {self.model_weights}")
        # TODO use a local cache to check for existing model weights (instead of re-downloading each time?)

        # confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thres

        # set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        logger.info(f"torch device: {device}")

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

        if not self.predictor:
            # load model...
            logger.info(f"Loading segmentation model {self.model_name}")
            self.predictor = DefaultPredictor(self.cfg)
            self.id_model = self._get_id_model(self.predictor.model)
            logger.info(f"Model ID: {self.id_model}")

        # --- run inference
        predictions = self.predictor(np.array(input.image))["instances"]
        predictions = predictions.to("cpu")

        results = []
        if not predictions:
            logger.warn("No segmentation predictions for this image!")
            return TaskResult(self._task_id)

        if (
            not predictions.has("scores")
            or not predictions.has("pred_classes")
            or not predictions.has("pred_masks")
        ):
            logger.warn(
                "Segmentation predictions are missing data or format is unexpected! Returning empty results"
            )
            return TaskResult(self._task_id)

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
                        id_model=self.id_model,
                    )
                    seg_results.append(seg_result)
        map_segmentation = MapSegmentation(doc_id=input.raster_id, segments=seg_results)
        return TaskResult(task_id=self._task_id, output=map_segmentation.model_dump())

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
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        return (res[-2], has_holes)

    def _get_id_model(self, model) -> str:
        """
        Create a unique string ID for this model,
        based on MD5 hash of the model's state-dict
        """

        state_dict_str = str(model.state_dict())
        hash_result = hashlib.md5(bytes(state_dict_str, encoding="utf-8"))
        return hash_result.hexdigest()
