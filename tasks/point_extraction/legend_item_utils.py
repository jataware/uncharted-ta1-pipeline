import logging
from enum import Enum
from typing import List, Tuple
from collections import defaultdict
from shapely import Polygon, distance

from tasks.point_extraction.entities import LegendPointItem, LegendPointItems
from tasks.point_extraction.label_map import LABEL_MAPPING, YOLO_TO_CDR_LABEL
from schema.cdr_schemas.cdr_responses.legend_items import LegendItemResponse
from tasks.segmentation.entities import MapSegmentation, SEGMENT_POINT_LEGEND_CLASS


logger = logging.getLogger(__name__)


# Legend item annotations "system" or provenance labels
class LEGEND_ANNOTATION_PROVENANCE(str, Enum):
    GROUND_TRUTH = "ground_truth"
    LABELME = "labelme"  # aka STEPUP
    POLYMER = "polymer"  # Jatware's annotation system

    def __str__(self):
        return self.value


def parse_legend_annotations(
    legend_anns: list,
    raster_id: str,
    system_filter=[LEGEND_ANNOTATION_PROVENANCE.POLYMER],
    check_validated=True,
) -> LegendPointItems:
    """
    parse legend annotations JSON data (CDR LegendItemResponse json format)
    and convert to LegendPointItem objects
    """

    # parse legend annotations and group by system label
    legend_item_resps = defaultdict(list)
    count_leg_items = 0
    for leg_ann in legend_anns:
        try:
            leg_resp = LegendItemResponse(**leg_ann)
            validated_ok = leg_resp.validated if check_validated else True
            if leg_resp.system in system_filter and validated_ok:
                # only keep legend item responses from desired systems
                legend_item_resps[leg_resp.system].append(leg_resp)
                count_leg_items += 1
        except Exception as e:
            # legend_pt_items = LegendPointItems(items=[], provenance="")
            logger.error(
                f"EXCEPTION parsing legend annotations json for raster {raster_id}: {repr(e)}"
            )
    logger.info(f"Successfully loaded {count_leg_items} LegendItemResponse objects")

    # try to parse non-labelme annotations first
    legend_point_items = []
    system_label = ""
    for system, leg_anns in legend_item_resps.items():
        if system == LEGEND_ANNOTATION_PROVENANCE.LABELME:
            continue
        system_label = system
        legend_point_items.extend(legend_ann_to_legend_items(leg_anns, raster_id))
    if legend_point_items:
        logger.info(f"Parsed {len(legend_point_items)} legend point items")
        return LegendPointItems(items=legend_point_items, provenance=system_label)
    else:
        # try to parse labelme annotations 2nd (since labelme anns have noisy data for point/line features)
        for system, leg_anns in legend_item_resps.items():
            if not system == LEGEND_ANNOTATION_PROVENANCE.LABELME:
                continue
            legend_point_items.extend(legend_ann_to_legend_items(leg_anns, raster_id))
        if legend_point_items:
            logger.info(f"Parsed {len(legend_point_items)} labelme legend point items")
            return LegendPointItems(
                items=legend_point_items,
                provenance=LEGEND_ANNOTATION_PROVENANCE.LABELME,
            )
    logger.info(f"Parsed 0 legend point items")
    return LegendPointItems(items=[], provenance="")


def parse_legend_point_hints(legend_hints: dict, raster_id: str) -> LegendPointItems:
    """
    parse legend hints JSON data (from the CMA contest)
    and convert to LegendPointItem objects

    legend_hints -- input hints dict
    """

    legend_point_items = []
    for shape in legend_hints["shapes"]:
        label = shape["label"]
        if not label.endswith("_pt") and not label.endswith("_point"):
            continue  # not a point symbol, skip

        # contour coords for the legend item's thumbnail swatch
        xy_pts = shape.get("points", [])
        if xy_pts:
            x_min = xy_pts[0][0]
            x_max = xy_pts[0][0]
            y_min = xy_pts[0][1]
            y_max = xy_pts[0][1]
            for x, y in xy_pts:
                x_min = int(min(x, x_min))
                x_max = int(max(x, x_max))
                y_min = int(min(y, y_min))
                y_max = int(max(y, y_max))
        else:
            x_min = 0
            x_max = 0
            y_min = 0
            y_max = 0
        xy_pts = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]
        class_name = find_legend_keyword_match(label, raster_id)
        legend_point_items.append(
            LegendPointItem(
                name=label,
                class_name=class_name,
                legend_bbox=[x_min, y_min, x_max, y_max],
                legend_contour=xy_pts,
                confidence=1.0,
                system=LEGEND_ANNOTATION_PROVENANCE.GROUND_TRUTH,
                validated=True,
            )
        )
    return LegendPointItems(
        items=legend_point_items, provenance=LEGEND_ANNOTATION_PROVENANCE.GROUND_TRUTH
    )


def find_legend_keyword_match(legend_item_name: str, raster_id: str) -> str:
    """
    Use keyword matching to map legend item label to point extractor ontology class names
    """
    leg_label_norm = raster_id + "_" + legend_item_name.strip().lower()
    matches = []
    for symbol_class, suffixs in LABEL_MAPPING.items():
        for s in suffixs:
            if s in leg_label_norm:
                # match found
                matches.append((s, symbol_class))
    if matches:
        # sort to get longest suffix match
        matches.sort(key=lambda a: len(a[0]), reverse=True)
        symbol_class = matches[0][1]
        logger.info(
            f"Legend label: {legend_item_name} matches point class: {symbol_class}"
        )
        return symbol_class

    # if no matches, then double-check exact matches with CDR ontology terms
    cdr_to_yolo = {v: k for k, v in YOLO_TO_CDR_LABEL.items()}
    leg_label_norm = legend_item_name.strip().lower()
    if leg_label_norm in cdr_to_yolo:
        # match found
        symbol_class = cdr_to_yolo[leg_label_norm]
        logger.info(
            f"Legend label: {legend_item_name} matches point class: {symbol_class}"
        )
        return symbol_class

    logger.info(f"No point class match found for legend label: {legend_item_name}")
    return ""


def get_swatch_contour(bbox: List, xy_pts: List[List]) -> Tuple:
    if bbox and xy_pts:
        return (bbox, xy_pts)
    if xy_pts:
        # calc bbox from contour
        p = Polygon(xy_pts)
        bbox = list(p.bounds)
    if bbox:
        xy_pts = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]
    return (bbox, xy_pts)


def legend_ann_to_legend_items(
    legend_anns: List[LegendItemResponse], raster_id: str
) -> List[LegendPointItems]:
    """
    convert LegendItemResponse (CDR schema format)
    to internal LegendPointItem objects
    """
    legend_point_items = []
    prev_label = ""
    for leg_ann in legend_anns:
        label = leg_ann.label if leg_ann.label else leg_ann.abbreviation
        if (
            leg_ann.system == LEGEND_ANNOTATION_PROVENANCE.LABELME
            and prev_label
            and prev_label == label
        ):
            # special base to handle labelme (STEPUP) annotations...
            # skip the 2nd labelme annotation in each pair
            # (this 2nd entry is just the bbox for the legend item text; TODO -- could extract and include this text too?)
            continue
        if (
            not leg_ann.system == LEGEND_ANNOTATION_PROVENANCE.LABELME
            and not leg_ann.category == "point"
        ):
            # this legend annotation does not represent a point feature; skip
            # (Note: LABELME annotations are handled separately, because it labels ALL feature types as "polygons")
            continue

        class_name = find_legend_keyword_match(label, raster_id)

        xy_pts = leg_ann.px_geojson.coordinates[0] if leg_ann.px_geojson else []
        (bbox, xy_pts) = get_swatch_contour(leg_ann.px_bbox, xy_pts)

        legend_point_items.append(
            LegendPointItem(
                name=label,
                class_name=class_name,
                abbreviation=leg_ann.abbreviation,
                description=leg_ann.description,
                legend_bbox=bbox,
                legend_contour=xy_pts,
                system=leg_ann.system,
                system_version=leg_ann.system_version,
                confidence=leg_ann.confidence,
                validated=leg_ann.validated,
            )
        )
        prev_label = label

    return legend_point_items


def filter_labelme_annotations(
    leg_point_items: LegendPointItems,
    segmentation: MapSegmentation,
    width_thres=120,
    shape_thres=2.0,
):
    """
    labelme (aka STEPUP) legend annotations are noisy, with all items for polygons, points and lines grouped together.
    These are filtered using segmentation info and shape heuristics to estimate which items, if any, correspond to point features
    """

    segs_point_legend = list(
        filter(
            lambda s: (s.class_label == SEGMENT_POINT_LEGEND_CLASS),
            segmentation.segments,
        )
    )
    if not segs_point_legend:
        logger.warning(
            "No Points-Legend segment found. Disregarding labelme legend annotations as noisy."
        )
        leg_point_items.items = []
        return

    filtered_leg_items = []
    for seg in segs_point_legend:
        p_seg = Polygon(seg.poly_bounds)
        for leg in leg_point_items.items:
            p_leg = Polygon(leg.legend_contour)
            if not p_seg.intersects(p_leg.centroid):
                # this legend swatch is not within the points legend area; disregard
                continue
            # legend swatch intersects the points legend area
            # check other properties to determine if swatch is line vs point symbol
            w = leg.legend_bbox[2] - leg.legend_bbox[0]
            h = leg.legend_bbox[3] - leg.legend_bbox[1]
            if leg.class_name:
                # legend item label is in points ontology
                filtered_leg_items.append(leg)
            elif w < width_thres and w < shape_thres * h:
                # legend item swatch bbox is close to square
                filtered_leg_items.append(leg)
    leg_point_items.items = filtered_leg_items


def legend_items_use_ontology(leg_point_items: LegendPointItems) -> bool:
    """
    Check if all legend items use the feature ontology
    (ie the class names are set)
    """
    class_labels_ok = True
    if len(leg_point_items.items) > 0:
        for leg_item in leg_point_items.items:
            if not leg_item.class_name:
                logger.info(
                    "Point ontology labels are missing for some of the legend items. Proceeding with tiling and further analysis of legend area..."
                )
                class_labels_ok = False
                break
        if class_labels_ok:
            logger.info(
                f"*** Point ontology labels are available for ALL legend items. Skipping further legend item analysis."
            )
    return class_labels_ok


def handle_duplicate_labels(leg_point_items: LegendPointItems):
    """
    De-duplicate legend item labels, if present
    """

    leg_labels = set()
    for leg_item in leg_point_items.items:
        if leg_item.name in leg_labels:
            suffix = 1
            while str(f"{leg_item.name}_{suffix}") in leg_labels and suffix < 99:
                suffix += 1
            dedup_name = f"{leg_item.name}_{suffix}"
            logger.info(
                f"Multiple legend items named {leg_item.name}; using {dedup_name} for duplicate item"
            )
            leg_item.name = dedup_name
            leg_labels.add(dedup_name)
        else:
            leg_labels.add(leg_item.name)
