from tasks.point_extraction.entities import MapPointLabel, MapImage
from tasks.common.task import Task, TaskInput, TaskResult

from typing import List
import parmap
import cv2
import numpy as np
from PIL import Image


class PointDirectionPredictor(Task):
    _VERSION = 1
    SUPPORTED_CLASSES = ["strike_and_dip"]
    TEMPLATE_PATH = "tasks/point_extraction/templates/strike_dip_implied_transparent_black_north.jpg"
    template: np.ndarray = np.empty((0, 0))

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def load_template(self):
        template = np.array(Image.open(self.TEMPLATE_PATH))
        self.template = self.global_thresholding(
            template, otsu=False, threshold_value=225
        )  # Blacken template.
        self.template = template

    def _check_size(self, label: MapPointLabel, template: np.ndarray) -> bool:
        """
        Used to palliate issues arising from tiling..
        Temporary fix which prevents propagating tiling errors to directionality pred.
        """
        x1, y1, x2, y2 = label.x1, label.y1, label.x2, label.y2
        template_size = template.shape[:2]
        label_size = (x2 - x1, y2 - y1)
        if label_size[0] < template_size[0] or label_size[1] < template_size[1]:
            return False
        return True

    @staticmethod
    def create_mask(template: np.ndarray, ignore_color=[255, 255, 255]):
        """
        Creates a binary mask where pixels equal to ignore_color are set to 0, and all others are set to 1.

        Parameters:
        - template (np.array): The template image.
        - ignore_color (list of int): The RGB color to ignore. Default is white, [255, 255, 255].

        Returns:
        np.array: The binary mask.
        """
        # Check if the image is grayscale
        if len(template.shape) == 2:
            # If it's grayscale, create a simple binary mask
            mask = np.ones_like(template, dtype=np.uint8) * 255
        else:
            # If it's not grayscale, create a mask ignoring a specific color
            mask = np.all(template != ignore_color, axis=-1).astype(np.uint8) * 255

        return mask

    @staticmethod
    def rotate_image(image: np.ndarray, angle, border_color=(255, 255, 255)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_theta = np.abs(M[0, 0])
        sin_theta = np.abs(M[0, 1])
        new_w = int(h * sin_theta + w * cos_theta)
        new_h = int(h * cos_theta + w * sin_theta)

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_color,
        )

        return rotated

    @staticmethod
    def crop_edges(img: np.ndarray, size=(70, 70)):
        h, w = img.shape[:2]
        h_crop, w_crop = size
        h_offset, w_offset = (h - h_crop) // 2, (w - w_crop) // 2
        return img[h_offset : h_offset + h_crop, w_offset : w_offset + w_crop]

    def match_template(self, base_image: np.ndarray, template: np.ndarray):
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        base_image = base_image.astype(np.float32)
        template = template.astype(np.float32)

        diag_length = int(
            np.ceil(np.sqrt(template.shape[0] ** 2 + template.shape[1] ** 2))
        )
        pad_y = max(diag_length - base_image.shape[0], base_image.shape[0])
        pad_x = max(diag_length - base_image.shape[1], base_image.shape[1])
        base_image = self.pad_image(
            base_image,
            pad_size=(base_image.shape[0] + pad_y, base_image.shape[1] + pad_x),
        )
        if len(base_image.shape) == 3:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_image = base_image.astype(np.float32)
        mask = self.create_mask(template)
        result = cv2.matchTemplate(base_image, template, cv2.TM_CCOEFF, mask=mask)
        return result

    @staticmethod
    def global_thresholding(img: np.ndarray, threshold_value=50, otsu=False):
        """
        Applies global thresholding to an image and displays the original and thresholded image.

        Parameters:
        - img (np.array): The input image in NumPy array format.
        - threshold_value (int): The global threshold value.
        """
        # Handle images with alpha channels.
        if len(img.shape) > 2 and img.shape[2] == 4:
            # Separate the alpha channel
            bgr, alpha = img[..., :3], img[..., 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        elif len(img.shape) > 2 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            alpha = None
        else:
            gray = img.copy()
            alpha = None

        thresh = cv2.THRESH_BINARY

        if otsu:
            thresh = cv2.THRESH_OTSU

        _, thresh_img = cv2.threshold(gray, threshold_value, 255, thresh)

        # If the image has an alpha channel, merge the thresholded image and the alpha channel
        if alpha is not None and img.shape[2] == 4:
            thresh_img = cv2.merge((thresh_img, thresh_img, thresh_img, alpha))

        return thresh_img

    def _fetch_template_match_candidates(
        self, labels: List[MapPointLabel], class_name="strike_and_dip"
    ):
        if class_name not in self.SUPPORTED_CLASSES:
            raise RuntimeError(
                f"Class {class_name} is not supported by this predictor."
            )
        return [(i, p) for i, p in enumerate(labels) if p.class_name == class_name]

    def _construct_base_candidates(self, labels: List[MapPointLabel], img: MapImage):
        return [
            np.array(img.image.crop((p.x1, p.y1, p.x2, p.y2)))
            for p in labels
            if self._check_size(p, template=self.template)
        ]

    def pad_image(self, img: np.ndarray, pad_size=(100, 100)):
        h, w = img.shape[:2]
        h_pad, w_pad = pad_size
        h_offset, w_offset = (h_pad - h) // 2, (w_pad - w) // 2

        if len(img.shape) == 3:
            num_channels = img.shape[2]
        else:
            num_channels = 1

        if num_channels == 3:
            padded_img = np.ones((h_pad, w_pad, 3), dtype=np.uint8) * 255
            padded_img[h_offset : h_offset + h, w_offset : w_offset + w, :] = img
        else:
            padded_img = np.ones((h_pad, w_pad), dtype=np.uint8) * 255
            padded_img[h_offset : h_offset + h, w_offset : w_offset + w] = img

        return padded_img

    def predict(self, base_image: np.ndarray):
        """
        Parameters:
        - template (np.array): The template image.
        - base_image (np.array): The base image.
        """
        if self.template.shape[0] == 0:
            self.load_template()

        base_image = self.global_thresholding(base_image)

        deg = []
        score = []
        best_score = -np.inf
        best_angle = 0

        for i in range(0, 360, 1):
            rotated_template = self.rotate_image(self.template[:, :], i)
            result = self.match_template(base_image, rotated_template)
            base_image_midpoint = np.array(base_image.shape[:2]) // 2
            result = result[base_image_midpoint[0] - 10 : base_image_midpoint[0] + 10]
            max_score = result.max()

            if max_score > best_score:
                best_score = max_score
                best_angle = i
            deg.append(i)
            score.append(max_score)

        return 360 - best_angle

    def run(self, input: TaskInput) -> TaskResult:
        """
        Run batch predictions over a MapImage object.

        This modifies the MapImage object predictions inplace. Unit test this.
        """

        if PointDirectionPredictor.template.shape[0] == 0:
            self.load_template()
        input_map_image = MapImage.model_validate(
            input.data["map_image"]
        )  # todo should just use the task input
        map_image = input_map_image.model_copy()
        if map_image.labels is None:
            raise RuntimeError("MapImage must have labels to run batch_predict.")
        match_candidates = self._fetch_template_match_candidates(
            map_image.labels
        )  # [[idx, label], ...] Use idx to add direction to labels.
        idxs = [idx for idx, _ in match_candidates]
        labels_with_direction = [label for _, label in match_candidates]
        base_images = self._construct_base_candidates(
            labels_with_direction, map_image
        )  # Crop map to regions of interest.
        direction_preds = parmap.map(self.predict, base_images, pm_pbar=True)
        for idx, direction in zip(idxs, direction_preds):
            map_image.labels[idx].directionality = direction
        return TaskResult(
            task_id=self._task_id, output={"map_image": map_image.model_dump()}
        )

    @property
    def input_type(self):
        return MapImage

    @property
    def output_type(self):
        return MapImage
