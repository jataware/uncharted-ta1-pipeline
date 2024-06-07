import logging
import math
from tasks.segmentation.entities import MapSegmentation
from shapely.geometry import Polygon
from typing import List

logger = logging.getLogger(__name__)


def segmenter_postprocess(segmentation: MapSegmentation, class_labels: List):
    """
    Post-process the segmentation result, by ranking segments per class
    """
    segments_out = []
    # loop through segmenter class labels, and de-noise / rank segments per class
    for label in class_labels:
        segments = list(
            filter(lambda s: (s.class_label == label), segmentation.segments)
        )
        if len(segments) == 0:
            continue
        # rank the segments by heuristic of confidence x sqrt(area)
        segments.sort(key=lambda s: (s.confidence * math.sqrt(s.area)), reverse=True)
        segments_out.extend(segments)

        # TODO -- could prune very low confidence segments and/or merge overlapping segments together

    # save ranked segments
    segmentation.segments = segments_out


def get_segment_bounds(
    segmentation: MapSegmentation, segment_class: str, max_results: int = 0
) -> List[Polygon]:
    """
    Parse segmentation result and return the polygon bounds for the desired segmentation class, if present
    Assumes the segments per class have already been ranked using confidence * sqrt(area) as in 'segmenter_postprocess'

    segment_class -- class to fetch
    max_results -- max segments to return
                if =0, all matching segments are returned
    """

    # filter segments for desired class
    segments = list(
        filter(lambda s: (s.class_label == segment_class), segmentation.segments)
    )
    if not segments:
        logger.warning(f"No {segment_class} segment found")
        return []
    if max_results > 0:
        segments = segments[:max_results]
    return [Polygon(s.poly_bounds) for s in segments]
