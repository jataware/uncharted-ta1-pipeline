import cv2
import logging
import math
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from collections import defaultdict

from tasks.segmentation.entities import (
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
    SEGMENT_MAP_CLASS,
)
from tasks.segmentation.segmenter_utils import get_segment_bounds, segments_to_mask
from tasks.point_extraction import point_extractor_utils as pe_utils
from tasks.common.task import Task, TaskInput, TaskResult

from tasks.text_extraction.entities import (
    DocTextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.point_extraction.entities import (
    PointLabels,
    PointLabel,
    LegendPointItem,
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
    MAP_PT_LABELS_OUTPUT_KEY,
)

MODEL_NAME = "uncharted_oneshot_point_extractor"
MODEL_VER = "0.0.1"

OCR_MIN_LEN = 3
CONTOUR_SIZE_FACTOR = 2.0
CONTOUR_THICKNESS = 2
MORPH_ITER = 2
WHITE_SHIFT_LINES = 65
ROTATE_INTERVAL = 23

# good range (0.05 to 0.2) -- higher = better recall, but worse precision
XCORR_DELTA = 0.17
# good range (0.5 to 0.7) -- lower = better recall, but worse precision
XCORR_MIN = 0.45
XCORR_MAX = 0.8

MATCH_DIST_FACTOR = 0.5

MAX_MATCHES = 250
MIN_MATCHES = 0  # if YOLO finds > min_matches, then skip template point extraction

logger = logging.getLogger(__name__)


class TemplateMatchPointExtractor(Task):
    """
    One-shot point extractor based on CV template-matching
    """

    def __init__(
        self,
        task_id: str,
        cache_path: str,
    ):

        super().__init__(task_id, cache_path)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run template match point symbol extractor
        """

        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
        else:
            legend_pt_items = LegendPointItems(items=[])

        if not legend_pt_items or not legend_pt_items.items:
            logger.info(
                "No Legend item info available. Skipping Template-Match Point Extractor"
            )
            result = self._create_result(task_input)
            result.add_output(
                MAP_PT_LABELS_OUTPUT_KEY, task_input.data[MAP_PT_LABELS_OUTPUT_KEY]
            )
            return result

        # get existing point predictions from YOLO point extractor
        if MAP_PT_LABELS_OUTPUT_KEY in task_input.data:
            map_point_labels = PointLabels.model_validate(
                task_input.data[MAP_PT_LABELS_OUTPUT_KEY]
            )
            if map_point_labels.labels is None:
                map_point_labels.labels = []
        else:
            map_point_labels = PointLabels(
                path="", raster_id=task_input.raster_id, labels=[]
            )

        # --- check which legend points still need to be processed, if any?
        pt_features = self._which_points_need_processing(
            map_point_labels.labels, legend_pt_items.items, min_predictions=MIN_MATCHES  # type: ignore
        )

        if not pt_features:
            # no template-matching point extraction needed
            logger.info(
                "No legend items need further processing. Skipping Template-Match Point Extractor"
            )
            result = self._create_result(task_input)
            result.add_output(
                MAP_PT_LABELS_OUTPUT_KEY, task_input.data[MAP_PT_LABELS_OUTPUT_KEY]
            )
            return result

        # --- check cache and re-use existing result if present
        doc_key = f"{task_input.raster_id}_templatematch_points-{MODEL_VER}"
        templatematch_point_labels = self._get_cached_data(doc_key, pt_features)
        if templatematch_point_labels:
            logger.info(
                f"Using cached template-match point extraction results for raster: {task_input.raster_id}"
            )
            map_point_labels.labels.extend(templatematch_point_labels.labels)  # type: ignore
            return TaskResult(
                task_id=self._task_id,
                output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
            )

        templatematch_point_labels = PointLabels(
            path="", raster_id=task_input.raster_id, labels=[]
        )

        # convert image from PIL to opencv (numpy) format --  assumed color channel order is RGB
        im_in = np.array(task_input.image)

        # get legend template images,
        # de-noise templates and create binary masks
        im_template_and_masks = [
            pe_utils.template_conncomp_denoise(x)
            for x in self._get_template_images(im_in, pt_features)
        ]

        # --- get OCR output
        img_text = (
            DocTextExtraction.model_validate(
                task_input.data[TEXT_EXTRACTION_OUTPUT_KEY]
            )
            if TEXT_EXTRACTION_OUTPUT_KEY in task_input.data
            else DocTextExtraction(doc_id=task_input.raster_id, extractions=[])
        )

        # --- mask OCR blocks
        im_in = pe_utils.mask_ocr_blocks(
            im_in,
            img_text.extractions,
            max_area=0,
            min_len=OCR_MIN_LEN,
            prune_symbols=True,
        )

        # --- get segment info for the map ROI
        map_roi = [0, 0, im_in.shape[1], im_in.shape[0]]
        if SEGMENTATION_OUTPUT_KEY in task_input.data:
            segmentation = MapSegmentation.model_validate(
                task_input.data[SEGMENTATION_OUTPUT_KEY]
            )
            # get a binary mask of the regions-of-interest and apply to the input image
            binary_mask = segments_to_mask(
                segmentation,
                (im_in.shape[1], im_in.shape[0]),
                roi_classes=[SEGMENT_MAP_CLASS],
            )
            if binary_mask.size != 0:
                # apply binary mask to input image
                im_in = cv2.bitwise_and(im_in, im_in, mask=binary_mask)
            p_map = get_segment_bounds(segmentation, SEGMENT_MAP_CLASS)
            if len(p_map) > 0:
                # restrict to use *only* the bounding rectangle of map area
                # TODO: ideally should use map polygon area as a binary mask
                p_map = p_map[0]  # use 1st (highest ranked) map segment
                map_roi = [int(b) for b in p_map.bounds]
                # crop image to the map ROI
                im_in = im_in[map_roi[1] : map_roi[3], map_roi[0] : map_roi[2], :]

        # --------------
        # loop through all available point legend items
        for i, pt_feature in enumerate(pt_features):
            feature_name = (
                pt_feature.class_name if pt_feature.class_name else pt_feature.name
            )
            logger.info(f"Processing {feature_name}")
            matches_dedup = []

            # --- pre-process the main image and template image, before template matching
            im, im_templ = pe_utils.image_pre_processing(
                im_in.copy(), im_template_and_masks[i][0], im_template_and_masks[i][1]
            )
            # note: im and im_templ are in RGB colour-space

            # convert to gray and get foregnd mask for template
            # TODO could also just crop/re-size the fore mask here too?
            templ_thres, fore_mask = cv2.threshold(
                cv2.cvtColor(im_templ, cv2.COLOR_RGB2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

            #
            # --- Use edges + contour analysis to extract long, continouous lines and de-emphasize them (ie fade to white)
            im_temp = im.copy()
            blurredImage = cv2.GaussianBlur(
                cv2.cvtColor(im_temp, cv2.COLOR_RGB2GRAY), (3, 3), 0
            )
            threshold, thresholdImage = cv2.threshold(
                blurredImage, 0, 255, cv2.THRESH_OTSU
            )
            im_temp = cv2.Canny(blurredImage, threshold, threshold / 2)
            contours, hierarchy = cv2.findContours(
                im_temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            contour_thres = (
                max(im_templ.shape) * CONTOUR_SIZE_FACTOR
            )  # find large contours to de-emphasize
            contours_large = []
            for cnt in contours:
                (xc, yc), rc = cv2.minEnclosingCircle(cnt)
                if 2 * rc > contour_thres:
                    contours_large.append(cnt)

            kernel_morph = np.ones((3, 3), np.uint8)
            im_large_contours = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            cv2.drawContours(im_large_contours, contours_large, -1, 255, CONTOUR_THICKNESS)  # type: ignore
            if MORPH_ITER > 0:
                im_large_contours = cv2.dilate(
                    im_large_contours, kernel_morph, iterations=MORPH_ITER
                )
            if MORPH_ITER > 1:
                im_large_contours = cv2.erode(
                    im_large_contours, kernel_morph, iterations=(MORPH_ITER - 1)
                )

            # ---- De-emphasize long, continuous lines by fading to white
            idx = im_large_contours > 0  # type: ignore # pxl x,y for large contours
            im = im.astype(np.float32)
            im[idx] = np.clip(im[idx] + WHITE_SHIFT_LINES, 0, 255)
            im = im.astype(np.uint8)

            # ---- Template matching
            # TODO -- re-factor and clean up dup code with point_orientation_extractor?
            # loop through rotational intervals...
            im_xcorr_all = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
            for rot_deg in range(0, 360, ROTATE_INTERVAL):
                logger.info("template rotation: {}".format(rot_deg))

                # get rotated template
                if rot_deg > 0:
                    template = ndimage.rotate(im_templ, rot_deg, cval=255)
                else:
                    template = im_templ.copy()
                # get rotated foregnd mask
                fore_mask_rot = (
                    ndimage.rotate(fore_mask, rot_deg, cval=0)
                    if rot_deg > 0
                    else fore_mask.copy()
                )

                # crop the rotated template
                template = pe_utils.crop_template(
                    template,
                    fore_mask_rot,
                    crop_buffer=5,
                )
                templ_hw = template.shape[0], template.shape[1]

                if rot_deg > 0 and self._is_template_redundant(im_templ, template):
                    # this template rotation is highly correlated with the original (unrotated) template
                    # skip -- is redundant to check for matches in this case
                    continue

                # ---- find all the template matches with base map
                im_xcorr = cv2.matchTemplate(
                    im,
                    template,
                    cv2.TM_CCOEFF_NORMED,
                )
                im_xcorr = np.nan_to_num(
                    im_xcorr, copy=False, nan=0.0, posinf=0.0, neginf=0.0
                )
                logger.info("max x-corr: {}".format(im_xcorr.max()))

                im_xcorr = np.clip((im_xcorr * 255), 0, 255).astype(np.uint8)  # type: ignore
                tx_half = int(template.shape[1] * 0.5)
                ty_half = int(template.shape[0] * 0.5)
                im_xcorr_all[
                    ty_half : (ty_half + im_xcorr.shape[0]),
                    tx_half : (tx_half + im_xcorr.shape[1]),
                ] = np.maximum(
                    im_xcorr,
                    im_xcorr_all[
                        ty_half : (ty_half + im_xcorr.shape[0]),
                        tx_half : (tx_half + im_xcorr.shape[1]),
                    ],
                )

            xcorr_max_val = im_xcorr_all.max()
            # dynamically set the x-correlation threshold
            xcorr_thres = int(
                max(
                    min(xcorr_max_val / 255.0, XCORR_MAX) * (1.0 - XCORR_DELTA),
                    XCORR_MIN,
                )
                * 255
            )
            logger.info("x-corr threshold {}".format(xcorr_thres / 255))

            templ_hw = (im_templ.shape[0], im_templ.shape[1])
            matches = self._find_match_candidates(im_xcorr_all, xcorr_thres, templ_hw)

            # ---- de-duplicate matches (prune overlapping template matches)
            templ_dim = float(templ_hw[0] + templ_hw[1]) / 2
            templ_dim_thres = math.pow(templ_dim * MATCH_DIST_FACTOR, 2)

            for x, y, tm_val in matches:
                if len(matches_dedup) >= MAX_MATCHES:
                    # we have enough good matches
                    break
                match_ok = True
                for mx, my, m_val in matches_dedup:
                    dist_sq = math.pow(x - mx, 2) + math.pow(y - my, 2)
                    if dist_sq < templ_dim_thres:
                        # discard - overlaps with other matches
                        match_ok = False
                        break
                if match_ok:
                    # append to existing matches
                    matches_dedup.append((x, y, tm_val))

            logger.info(
                "Final number of unique points extracted for label {}: {}".format(
                    feature_name, len(matches_dedup)
                )
            )
            if len(matches_dedup) > 0:
                preds = self._process_output(matches_dedup, feature_name, map_roi)
                templatematch_point_labels.labels.extend(preds)  # type: ignore

        # write to cache (note: only cache the template matched point extractions; not all of them)
        self.write_result_to_cache(templatematch_point_labels.model_dump(), doc_key)

        # append template-match extractions to main point extractions results
        map_point_labels.labels.extend(templatematch_point_labels.labels)  # type: ignore
        return TaskResult(
            task_id=self._task_id,
            output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
        )

    def _get_template_images(
        self, im: np.ndarray, feats: List[LegendPointItem]
    ) -> List[np.ndarray]:

        im_templates = []
        for feat in feats:
            (xmin, ymin, xmax, ymax) = [int(a) for a in feat.legend_bbox]
            im_templ = (im[ymin:ymax, xmin:xmax]).copy()
            im_templates.append(im_templ)

        return im_templates

    def _find_match_candidates(
        self, im_xcorr: np.ndarray, xcorr_thres: int, templ_hw: Tuple
    ) -> List:

        matches = []
        loc = np.where(im_xcorr >= xcorr_thres)
        loc_values = im_xcorr[loc]

        logger.debug("Num found pts: {}".format(len(loc[0])))

        # convert results to a list of tuples and sort
        matches = []
        for y, x, tm_val in zip(*loc, loc_values):
            cx = x
            cy = y

            matches.append((cx, cy, tm_val))

        # sort matches by correlation value (descending)
        matches.sort(key=lambda tup: tup[2], reverse=True)

        return matches

    def _is_template_redundant(
        self, templ_orig: np.ndarray, templ_rot: np.ndarray
    ) -> bool:

        h = min(templ_orig.shape[0], templ_rot.shape[0])
        w = min(templ_orig.shape[1], templ_rot.shape[1])

        templ_xcorr = cv2.matchTemplate(
            templ_rot, templ_orig[0:h, 0:w], cv2.TM_CCOEFF_NORMED
        )
        templ_xcorr_max = templ_xcorr.max()
        if templ_xcorr_max > 0.65:
            return True
        return False

    def _is_template_valid(self, legend_item: LegendPointItem) -> bool:
        """
        Check if a Legend Item contains a valid template swatch
        """
        if not legend_item.legend_contour:
            # no legend contour
            return False
        bbox = legend_item.legend_bbox
        if not bbox or len(bbox) < 4:
            # invalid bbox
            return False
        if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
            # legend bbox has 0 area
            return False

        return True

    def _process_output(
        self,
        matches: List,
        label: str,
        map_roi: List[int],
        bbox_size: int = 90,
    ) -> List[PointLabel]:
        """
        Convert template based detection results
        to a list of PointLabel objects
        """
        pt_labels = []
        bbox_half = bbox_size / 2
        for x, y, xcorr in matches:
            # generate bboxes around center pt (arbitrary size)
            if map_roi:
                x += map_roi[0]
                y += map_roi[1]
            x1 = max(int(x - bbox_half), 0)
            y1 = max(int(y - bbox_half), 0)
            x2 = int(x + bbox_half)
            y2 = int(y + bbox_half)
            # prepare final result label
            # note: using hash(label) as class numeric ID
            pt_labels.append(
                PointLabel(
                    classifier_name=MODEL_NAME,
                    classifier_version=MODEL_VER,
                    class_id=hash(label),
                    class_name=label,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=xcorr / 255.0,
                )
            )

        return pt_labels

    def _which_points_need_processing(
        self,
        map_point_labels: List[PointLabel],
        legend_pt_items: List[LegendPointItem],
        min_predictions=0,
    ) -> List[LegendPointItem]:
        """
        Check which legend items still need processing
        (Since some point classes may've already been handled by the ML point extractor)
        """
        if min_predictions < 0:
            # process all legend items regardless of upstream ML predictions
            return legend_pt_items

        legend_items_unprocessed = []
        preds_per_class = defaultdict(int)
        for pred in map_point_labels:
            if pred.class_name:
                preds_per_class[pred.class_name] += 1

        for legend_item in legend_pt_items:

            if not self._is_template_valid(legend_item):
                logger.warning(
                    f"No valid legend template is available for legend item {legend_item.name} - skipping"
                )
                continue

            leg_item_name = (
                legend_item.class_name if legend_item.class_name else legend_item.name
            )
            if not leg_item_name:
                logger.warning(
                    f"No valid legend name is available for legend item - skipping"
                )
                continue

            if leg_item_name not in preds_per_class:
                # no ML predictions for this point type; needs processing
                legend_items_unprocessed.append(legend_item)
            elif preds_per_class[leg_item_name] < min_predictions:
                # only a few ML predictions for this point type; still needs processing
                legend_items_unprocessed.append(legend_item)

        return legend_items_unprocessed

    def _get_cached_data(
        self, doc_key: str, legend_items: List[LegendPointItem]
    ) -> Optional[PointLabels]:

        try:
            json_data = self.fetch_cached_result(doc_key)
            if json_data:
                templatematch_point_labels = PointLabels(**json_data)
                # check that the expected set of point types are in cache
                pt_types_legend = set()
                for leg in legend_items:
                    name = leg.class_name if leg.class_name else leg.name
                    pt_types_legend.add(name)
                pt_types_cache = set()
                if templatematch_point_labels.labels:
                    for pt_label in templatematch_point_labels.labels:
                        pt_types_cache.add(pt_label.class_name)
                if len(pt_types_legend - pt_types_cache) == 0:
                    # pt type sets are the same; cached data is ok
                    return templatematch_point_labels

        except Exception as e:
            logger.warning(f"Exception fetching cached data: {repr(e)}")
        return None
