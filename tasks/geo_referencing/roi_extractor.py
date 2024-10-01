import logging

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon
from skimage.filters.rank import entropy
from skimage.morphology import disk

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY

from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("roi_extractor")

ROI_PXL_LIMIT = 2000
SLICE_PERCENT = 0.05


def buffer_fixed(vertices: List[Tuple[float, float]], input: TaskInput) -> float:
    return 150


def buffer_image_ratio(vertices: List[Tuple[float, float]], input: TaskInput) -> float:
    return max(input.image.size) * 0.03


def buffer_roi_ratio(vertices: List[Tuple[float, float]], input: TaskInput) -> float:
    xs = [v[0] for v in vertices]
    x_size = max(xs) - min(xs)
    ys = [v[1] for v in vertices]
    y_size = max(ys) - min(ys)

    return max(x_size, y_size) * 0.05


class ROIExtractor(Task):
    def run(self, input: TaskInput) -> TaskResult:

        roi = self._extract_roi(input)

        result = super()._create_result(input)
        if roi is not None:
            result.output["roi"] = roi[0]
            result.output["roi_inner"] = roi[1]
        return result

    def _extract_roi(
        self, input: TaskInput
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        return ([], [])


class ModelROIExtractor(ROIExtractor):
    """
    Use the map segmentation result, and buffer (inwards and outwards) to create a ring ROI
    """

    _buffering_func: Callable

    def __init__(self, task_id: str, buffering_func: Callable):
        super().__init__(task_id)
        self._buffering_func = buffering_func

    def _extract_roi(
        self, input: TaskInput
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        # read segmentation output
        segmentation_output: MapSegmentation = input.parse_data(
            SEGMENTATION_OUTPUT_KEY, MapSegmentation.model_validate
        )

        # extract the polygon (use the first valid map segment)
        poly_raw = None
        conf = 0
        for sr in segmentation_output.segments:
            if sr.class_label == "map" and sr.confidence > conf:
                poly_raw = sr.poly_bounds
                conf = sr.confidence
                break
        if poly_raw is None:
            return None

        # expand the polygon outward
        buffer_size = self._buffering_func(poly_raw, input)
        logger.info(f"buffering roi by {buffer_size}")

        polygon = Polygon(poly_raw)
        buffered = polygon.buffer(buffer_size, join_style=2)  # type: ignore

        # shrink the polygon inward for the inner boundary
        polygon = Polygon(poly_raw)
        buffered_inner = polygon.buffer(-3 * buffer_size, join_style=2)  # type: ignore

        w = input.image.width
        h = input.image.height
        bufferred_coords = list(
            map(
                lambda x: self._limit_polygon(x, (0, 0), (w, h)),
                buffered.exterior.coords,
            )
        )
        buffered_inner_coords = list(
            map(
                lambda x: self._limit_polygon(x, (0, 0), (w, h)),
                buffered_inner.exterior.coords,
            )
        )

        return bufferred_coords, buffered_inner_coords

    def _limit_polygon(
        self,
        coord: Tuple[float, float],
        lower_limit: Tuple[float, float],
        upper_limit: Tuple[float, float],
    ) -> Tuple[float, float]:
        return (
            min(max(lower_limit[0], coord[0]), upper_limit[0]),
            min(max(lower_limit[1], coord[1]), upper_limit[1]),
        )


class EntropyROIExtractor(ROIExtractor):
    _entropy_thres_buffer = 0.1

    def __init__(self, task_id: str, _entropy_thres_buffer: float = 0.1):
        super().__init__(task_id)
        self._entropy_thres_buffer = _entropy_thres_buffer

    def _extract_roi(
        self, input: TaskInput
    ) -> Optional[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]]:
        # convert PIL image to opencv
        image = cv.cvtColor(np.array(input.image), cv.COLOR_RGB2BGR)
        # ocr_blocks = input.get_data("ocr_blocks")
        ocr_blocks: DocTextExtraction = input.parse_data(
            TEXT_EXTRACTION_OUTPUT_KEY, DocTextExtraction.model_validate
        )

        # mask text blocks with median pixel values
        # (so entropy analysis focuses on non-text image features)
        if ocr_blocks:
            for blk in ocr_blocks.extractions:
                xmin = blk.bounds[0].x
                xmax = blk.bounds[0].x
                ymin = blk.bounds[0].y
                ymax = blk.bounds[0].y
                for v in blk.bounds:
                    xmin = min(xmin, v.x)
                    xmax = max(xmax, v.x)
                    ymin = min(ymin, v.y)
                    ymax = max(ymax, v.y)
                if xmin == xmax or ymin == ymax:
                    continue
                # mask text by replacing text block with its median pixel value
                ocr_pxl_slice = image[
                    ymin:ymax, xmin:xmax, :
                ]  # TODO - or just get median val along the top?
                if ocr_pxl_slice.size > 0:
                    med_val = np.median(ocr_pxl_slice, axis=[0, 1])  # type: ignore
                    ocr_pxl_slice[:, :, 0] = med_val[0]
                    ocr_pxl_slice[:, :, 1] = med_val[1]
                    ocr_pxl_slice[:, :, 2] = med_val[2]

        # resize image - for faster entropy analysis
        im_resize_ratio = min(ROI_PXL_LIMIT / max(image.shape), 1.0)
        if im_resize_ratio < 1.0:
            width = int(image.shape[1] * im_resize_ratio)
            height = int(image.shape[0] * im_resize_ratio)
            image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # ---- entropy analysis of grayscale image
        entr_img = entropy(
            image, disk(5)
        )  # image[:,:,0]   # TODO - disk size percentage of image size? (smaller is faster)
        entr_img = entr_img / entr_img.max()  # normalize entropy values to 0 - 1.0
        entr_img = entr_img.astype(np.float32)

        avg_entropy = entr_img.mean()  # average entropy of whole image

        entropy_thres = avg_entropy - self._entropy_thres_buffer

        # get entropy stats of slices of the image (along x and y axes)
        y_e_avgs, y_vals = self._get_slice_avgs(entr_img, axis=0, s_width=SLICE_PERCENT)
        x_e_avgs, x_vals = self._get_slice_avgs(entr_img, axis=1, s_width=SLICE_PERCENT)

        # find region-of-interest bounds by excluding slices
        # with low entropy
        roi_y = self._find_roi_bounds(
            y_e_avgs, y_vals, entropy_thres, entr_img.shape[0]
        )
        roi_x = self._find_roi_bounds(
            x_e_avgs, x_vals, entropy_thres, entr_img.shape[1]
        )

        if im_resize_ratio < 1.0:
            roi_x[0] = int(roi_x[0] / im_resize_ratio)
            roi_x[1] = int(roi_x[1] / im_resize_ratio)
            roi_y[0] = int(roi_y[0] / im_resize_ratio)
            roi_y[1] = int(roi_y[1] / im_resize_ratio)

        return (
            [
                (roi_x[0], roi_y[0]),
                (roi_x[1], roi_y[0]),
                (roi_x[1], roi_y[1]),
                (roi_x[0], roi_y[1]),
            ],
            [],
        )

    def _get_slice_avgs(
        self, img: np.ndarray, axis: int = 0, s_width: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        if axis != 0 and axis != 1:
            logger.error("ERROR! not supported!")
        len0 = img.shape[axis]
        p_shift = int(s_width * len0)
        p1 = 0
        p2 = p_shift
        is_ok = True
        avg_vals = []
        p_vals = []
        while is_ok:
            if p2 > len0:
                p2 = len0
                is_ok = False
            if p1 >= p2:
                is_ok = False
                break
            s_img = img[p1:p2, :] if axis == 0 else img[:, p1:p2]
            if s_img.size > 0:
                avg_vals.append(s_img.mean())
                p_vals.append((p1 + p2) / 2)
                p1 = p2
                p2 += p_shift

        return avg_vals, p_vals

    def _find_roi_bounds(
        self, avg_vals: List[float], p_vals: List[float], e_thres: float, p_len: int
    ) -> List[int]:
        # find region of interest bounds (starting from bottem or right first,
        # then top or left)
        buffer = (
            p_len * 0.01
        )  # TODO could add a buffer here (based on full image dimension or ??)
        roi_p = [0, p_len]
        idx = len(avg_vals) - 1
        p_lim = p_len / 2  # retain at least 50% of image for region of interest
        p_prev = p_len - 1
        for e_val in reversed(avg_vals):
            if e_val > e_thres:
                roi_p[1] = int(p_prev + buffer)
                p_lim = max(p_vals[idx] - p_lim, 0)
                break
            if p_vals[idx] <= p_lim:
                roi_p[1] = int(p_prev + buffer)
                p_lim = max(p_vals[idx] - p_lim, 0)
                break
            p_prev = p_vals[idx]
            idx -= 1

        idx = 0
        p_prev = 0
        for entr in avg_vals:
            if entr > e_thres:
                roi_p[0] = int(p_prev - buffer)
                break
            if p_vals[idx] >= p_lim:
                roi_p[0] = int(p_prev - buffer)
                break
            p_prev = p_vals[idx]
            idx += 1

        roi_p[0] = max(roi_p[0], 0)
        roi_p[1] = min(roi_p[1], p_len - 1)

        return roi_p
