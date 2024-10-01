from tasks.point_extraction.entities import PointLabels, MAP_PT_LABELS_OUTPUT_KEY
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.point_extraction.label_map import POINT_CLASS
from tasks.point_extraction.task_config import PointOrientationConfig
from tasks.point_extraction import point_extractor_utils
from tasks.text_extraction.entities import (
    TextExtraction,
    DocTextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)

from typing import Dict, List, Optional
import cv2
import logging
import math
import numpy as np
import re
from PIL import Image
from collections import defaultdict
from shapely.geometry import box
from shapely.strtree import STRtree
from shapely import distance
from scipy import ndimage


logger = logging.getLogger(__name__)

RE_NONNUMERIC = re.compile(r"[^0-9]")  # matches non-numeric chars
CODE_VER = "0.0.2"

FOREGND_COLOR_LAB = [0, 128, 128]  # template default foregnd color (in LAB space)


class PointOrientationExtractor(Task):

    # ---- supported point classes and corresponding template image paths
    POINT_TEMPLATES = {
        str(
            POINT_CLASS.STRIKE_AND_DIP
        ): "tasks/point_extraction/templates/strike_dip_black_on_white_north_synthetic.png",
        str(
            POINT_CLASS.OVERTURNED_BEDDING
        ): "tasks/point_extraction/templates/overturned_bedding_pt_north.png",
        str(
            POINT_CLASS.VERTICAL_BEDDING
        ): "tasks/point_extraction/templates/vertical_bedding_pt_north.png",
        str(
            POINT_CLASS.INCLINED_FOLIATION
        ): "tasks/point_extraction/templates/inclined_foliation_pt_north.png",
        str(
            POINT_CLASS.INCLINED_FOLIATION_IGNEOUS
        ): "tasks/point_extraction/templates/inclined_foliation_igneous_pt_north.png",
        str(
            POINT_CLASS.VERTICAL_FOLIATION
        ): "tasks/point_extraction/templates/vertical_foliation_pt_north.png",
        str(
            POINT_CLASS.VERTICAL_JOINT
        ): "tasks/point_extraction/templates/vertical_joint_pt_north.png",
        str(
            POINT_CLASS.MINE_TUNNEL
        ): "tasks/point_extraction/templates/mine_tunnel_pt_north.png",
        str(
            POINT_CLASS.LINEATION
        ): "tasks/point_extraction/templates/lineation_pt_north.png",
    }

    # ---- task config per point class
    POINT_CONFIGS = {
        str(POINT_CLASS.STRIKE_AND_DIP): PointOrientationConfig(
            point_class=str(POINT_CLASS.STRIKE_AND_DIP), mirroring_correction=True
        ),
        str(POINT_CLASS.OVERTURNED_BEDDING): PointOrientationConfig(
            point_class=str(POINT_CLASS.OVERTURNED_BEDDING), mirroring_correction=True
        ),
        str(POINT_CLASS.VERTICAL_BEDDING): PointOrientationConfig(
            point_class=str(POINT_CLASS.VERTICAL_BEDDING),
            dip_number_extraction=False,
            rotate_max=180,
        ),
        str(POINT_CLASS.INCLINED_FOLIATION): PointOrientationConfig(
            point_class=str(POINT_CLASS.INCLINED_FOLIATION), mirroring_correction=True
        ),
        str(POINT_CLASS.INCLINED_FOLIATION_IGNEOUS): PointOrientationConfig(
            point_class=str(POINT_CLASS.INCLINED_FOLIATION_IGNEOUS),
            mirroring_correction=True,
        ),
        str(POINT_CLASS.VERTICAL_FOLIATION): PointOrientationConfig(
            point_class=str(POINT_CLASS.VERTICAL_FOLIATION),
            dip_number_extraction=False,
            rotate_max=180,
        ),
        str(POINT_CLASS.VERTICAL_JOINT): PointOrientationConfig(
            point_class=str(POINT_CLASS.VERTICAL_JOINT),
            dip_number_extraction=False,
            rotate_max=180,
        ),
        str(POINT_CLASS.MINE_TUNNEL): PointOrientationConfig(
            point_class=str(POINT_CLASS.MINE_TUNNEL),
            dip_number_extraction=False,
        ),
        str(POINT_CLASS.LINEATION): PointOrientationConfig(
            point_class=str(POINT_CLASS.LINEATION), mirroring_correction=False
        ),
    }

    def __init__(self, task_id: str, points_model_id: str, cache_path: str):
        self.points_model_id = points_model_id
        self.templates = self._load_templates()

        super().__init__(task_id, cache_path)

    def _load_templates(self) -> Dict:
        """
        Load template image for all supported point symbol types
        """
        templates = {}
        for point_class, template_path in self.POINT_TEMPLATES.items():
            templates[point_class] = np.array(Image.open(template_path))
        return templates

    def _dip_magnitude_extraction(
        self,
        matches: List,
        ocr_polygon_index: STRtree,
        text_extractions: List[TextExtraction],
    ) -> Dict:
        """
        Extract dip magnitudes for point symbols, based on nearby OCR text labels
        """

        dip_magnitudes = {}  # point symbol idx -> (extracted dip angle, (x,y centroid))

        for pt_idx, map_pt_label in matches:
            # bbox around point location
            xy_box = box(
                map_pt_label.x1, map_pt_label.y1, map_pt_label.x2, map_pt_label.y2
            )
            # find text blocks that intersect with point's bbox
            hits = ocr_polygon_index.query(xy_box, predicate="intersects")  #'overlaps')
            if hits.size > 0:
                matches_text = []
                matches_idx = []
                for idx in hits.flat:
                    ocr_text_match = RE_NONNUMERIC.sub(
                        "", text_extractions[idx].text
                    ).strip()
                    if ocr_text_match:
                        # 1 or 2 digit numbers only!
                        ocr_text_match = float(ocr_text_match[:2])
                        if ocr_text_match > 90:
                            # extracted dip magnitude must be <= 90 degrees
                            continue
                        matches_text.append(ocr_text_match)
                        matches_idx.append(idx)
                        # break
                if len(matches_idx) == 1:
                    this_poly = ocr_polygon_index.geometries.item(matches_idx[0])
                    dip_magnitudes[pt_idx] = (
                        matches_text[0],
                        (this_poly.centroid.x, this_poly.centroid.y),
                    )

                elif len(matches_idx) > 1:
                    # multiple matches! choose the one closest to symbol centroid!
                    min_dist = 999999999.0  # init large
                    min_idx = 0
                    for ii, m_idx in enumerate(matches_idx):
                        this_dist = distance(
                            xy_box.centroid,
                            ocr_polygon_index.geometries.item(m_idx).centroid,
                        )
                        if this_dist < min_dist:
                            min_dist = this_dist
                            min_idx = ii
                    this_poly = ocr_polygon_index.geometries.item(matches_idx[min_idx])
                    dip_magnitudes[pt_idx] = (
                        matches_text[min_idx],
                        (this_poly.centroid.x, this_poly.centroid.y),
                    )
        return dip_magnitudes

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Run batch predictions over a PointLabels object.

        This modifies the PointLabels object predictions in-place.
        """

        # get result from point extractor task (with point symbol predictions)
        map_point_labels = PointLabels.model_validate(
            task_input.data[MAP_PT_LABELS_OUTPUT_KEY]
        )
        if map_point_labels.labels is None:
            raise RuntimeError("PointLabels must have labels to run batch_predict")
        if len(map_point_labels.labels) == 0:
            logger.warning(
                "No point symbol extractions found. Skipping Point orientation extraction."
            )
            return TaskResult(
                task_id=self._task_id,
                output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
            )

        # --- check cache and re-use existing result if present
        doc_key = (
            f"{task_input.raster_id}_orientations-{self.points_model_id}-{CODE_VER}"
        )
        cached_point_labels = self._get_cached_data(
            doc_key, len(map_point_labels.labels)
        )
        if cached_point_labels:
            logger.info(
                f"Using cached orientation results for raster: {task_input.raster_id}"
            )
            return TaskResult(
                task_id=self._task_id,
                output={MAP_PT_LABELS_OUTPUT_KEY: cached_point_labels.model_dump()},
            )

        # --- get OCR output
        img_text = (
            DocTextExtraction.model_validate(
                task_input.data[TEXT_EXTRACTION_OUTPUT_KEY]
            )
            if TEXT_EXTRACTION_OUTPUT_KEY in task_input.data
            else DocTextExtraction(doc_id=task_input.raster_id, extractions=[])
        )

        if len(img_text.extractions) == 0:
            logger.warning(
                "Skipping extraction of dip magnitudes - no OCR data available."
            )

        # --- build OCR tree index
        ocr_polygon_index = point_extractor_utils.build_ocr_index(img_text.extractions)

        # --- group point extractions by class label
        supported_classes = list(self.POINT_TEMPLATES.keys())
        match_candidates = defaultdict(list)  # class name -> list of tuples
        for i, p in enumerate(map_point_labels.labels):
            # tuple of (original extraction id, pt extraction object)
            match_candidates[p.class_name].append((i, p))

        # --- image pre-processing
        im_preproc = point_extractor_utils.image_pre_processing(
            np.array(task_input.image), FOREGND_COLOR_LAB
        )

        # --- perform symbol orientation analysis for each point class...
        for c in supported_classes:
            if c not in match_candidates:
                # no points found for this class
                continue
            matches = match_candidates[c]
            task_config = self.POINT_CONFIGS[c]  # task config for this point class
            bbox_half = int(task_config.bbox_size / 2)
            bbox_size = bbox_half * 2  # (bbox_half = int so bbox_size must = even num)
            logger.info(
                f"Performing point orientation analysis for {len(matches)} point symbols of class {c}"
            )

            # ---- 1. extract dip magnitude per point (from nearby OCR text)
            dip_magnitudes = {}
            if task_config.dip_number_extraction:
                dip_magnitudes = self._dip_magnitude_extraction(
                    matches, ocr_polygon_index, img_text.extractions
                )
                logger.info(
                    f"Dip magntiudes extracted for {len(dip_magnitudes)} / {len(matches)} points"
                )
                # save dip angle results for this point class
                for idx, (dip_angle, _) in dip_magnitudes.items():
                    map_point_labels.labels[idx].dip = dip_angle

            # --- 2. estimate symbol orientation (using template matching)
            # pre-process template image before template matching
            im_templ, _ = point_extractor_utils.template_pre_processing(
                self.templates[c], np.array([])
            )

            # convert to gray and get foregnd mask for template
            _, fore_mask = cv2.threshold(
                cv2.cvtColor(im_templ, cv2.COLOR_RGB2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )  # TODO could also just crop/re-size the fore mask here too?

            # --- template matching
            # loop through rotational intervals...
            xcorr_results = {}
            for rot_deg in range(
                0, task_config.rotate_max, task_config.rotate_interval
            ):
                logger.debug("template rotation: {}".format(rot_deg))
                # get rotated template
                if rot_deg > 0:
                    im_templ_rot = ndimage.rotate(im_templ, rot_deg, cval=255)
                else:
                    im_templ_rot = im_templ.copy()
                # get rotated foregnd mask
                fore_mask_rot = (
                    ndimage.rotate(fore_mask, rot_deg, cval=0)
                    if rot_deg > 0
                    else fore_mask.copy()
                )
                # crop rotated template
                im_templ_rot = point_extractor_utils.crop_template(
                    im_templ_rot, fore_mask_rot, crop_buffer=5
                )

                if (
                    im_templ_rot.shape[0] > bbox_size
                    or im_templ_rot.shape[1] > bbox_size
                ):
                    # template image cannot be larger than candidate image swatch (will cause an opencv exception)
                    h = min(im_templ_rot.shape[0], bbox_size)
                    w = min(im_templ_rot.shape[1], bbox_size)
                    # TODO ideally this slice should be centered!
                    im_templ_rot = im_templ_rot[0:h, 0:w]

                # --- loop through all point locations and do template matching for this angle...
                for pt_idx, map_pt_label in matches:
                    # get thumbnail image around predicted point symbol
                    xc = int((map_pt_label.x2 + map_pt_label.x1) / 2)
                    yc = int((map_pt_label.y2 + map_pt_label.y1) / 2)
                    im_thumbnail = im_preproc[
                        yc - bbox_half : yc + bbox_half, xc - bbox_half : xc + bbox_half
                    ]

                    max_val, max_idx = point_extractor_utils.template_matching(
                        im_thumbnail, im_templ_rot, task_config.xcorr_search_range
                    )

                    # update the 'best' orientation match for this point symbol location
                    if pt_idx in xcorr_results:
                        if max_val > xcorr_results[pt_idx][0]:
                            xcorr_results[pt_idx] = (max_val, rot_deg)
                    else:
                        xcorr_results[pt_idx] = (max_val, rot_deg)
            # finished checking all angles

            if task_config.mirroring_correction and dip_magnitudes:
                # --- correct x-corr results based on OCR label position (to prevent 180-deg mirror confusion)...
                for pt_idx, map_pt_label in matches:
                    if pt_idx in dip_magnitudes:
                        # get center location of dip label associated with this point
                        xc_ocr, yc_ocr = dip_magnitudes[pt_idx][1]
                        # get center location of point symbol
                        xc = int((map_pt_label.x2 + map_pt_label.x1) / 2)
                        yc = int((map_pt_label.y2 + map_pt_label.y1) / 2)
                        # get current point orientation angle
                        (max_val, rot_deg) = xcorr_results[pt_idx]

                        xc_ocr -= xc
                        yc_ocr -= yc
                        yc_ocr *= -1
                        ocr_deg = math.atan2(yc_ocr, xc_ocr) * 180 / math.pi
                        if ocr_deg < 0:
                            ocr_deg += 360
                        if not point_extractor_utils.angle_in_range(
                            ocr_deg, rot_deg, rot_deg + 180
                        ):
                            # possible orientation mirroring confusion!
                            rot_deg_corr = rot_deg - 180
                            if rot_deg_corr < 0:
                                rot_deg_corr += 360
                            xcorr_results[pt_idx] = (max_val, rot_deg_corr)
                            logger.debug(
                                f"Correcting angle for symbol index {pt_idx}, was {rot_deg}, now {rot_deg_corr}"
                            )

            # save "best orientation angle" results for this point class
            for idx, (_, best_angle) in xcorr_results.items():
                # convert final result from 'trig' angle convention
                # to compass angle convention (CW with 0 deg at top)
                map_point_labels.labels[idx].direction = self._trig_to_compass_angle(
                    best_angle, task_config.rotate_max
                )

        # write to cache
        self.write_result_to_cache(map_point_labels, doc_key)

        return TaskResult(
            task_id=self._task_id,
            output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
        )

    def _trig_to_compass_angle(self, angle_deg: int, rotate_max: int) -> int:
        """
        Convert "trigonometry" angle (CCW with 0 deg to the right)
        to "compass" angle convention (CW with 0 at the top).
        NOTE: a symbol's "dip marker" (if applicable) points to the top when a symbol is in the 0 deg compass direction
        """
        angle_compass = -angle_deg
        if angle_compass < 0:
            angle_compass += 360
        if angle_compass > rotate_max:
            angle_compass -= rotate_max
        return angle_compass

    def _get_cached_data(
        self, doc_key: str, num_predictions: int
    ) -> Optional[PointLabels]:

        try:
            cached_data = self.fetch_cached_result(doc_key)
            if cached_data:
                map_point_labels = PointLabels(**cached_data)
                if (
                    map_point_labels.labels
                    and len(map_point_labels.labels) == num_predictions
                ):
                    # cached data is ok
                    return map_point_labels

        except Exception as e:
            logger.warning(
                f"Exception fetching cached data: {repr(e)}; disregarding cached point orientations for this raster"
            )
        return None

    @property
    def input_type(self):
        return PointLabels

    @property
    def output_type(self):
        return PointLabels
