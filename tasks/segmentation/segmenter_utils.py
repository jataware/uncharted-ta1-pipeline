import logging
from tasks.segmentation.entities import MapSegmentation
from shapely.geometry import Polygon
from typing import List

logger = logging.getLogger(__name__)


def get_segment_bounds(
    segmentation: MapSegmentation, segment_class: str, max_results: int = 1
) -> List[Polygon]:
    """
    Parse segmentation result and return the polygon bounds for the desired segmentation class, if present

    segment_class -- class to fetch
    max_results -- max segments to return - sorted by heuristic of (confidence * area), highest to lowest,
                if =0, all matching segments are returned
    """
    # TODO -- should be improved to better handle noisy results with overlapping segments

    # filter segments for desired class
    segments = list(
        filter(lambda s: (s.class_label == segment_class), segmentation.segments)
    )

    if not segments:
        logger.warning(f"No {segment_class} segment found")
        return []

    # sort results by heuristic of (confidence * area) and return top 'max_results' polygon bounds
    segments.sort(key=lambda s: (s.confidence * s.area), reverse=True)
    if max_results > 0:
        segments = segments[:max_results]
    return [Polygon(s.poly_bounds) for s in segments]
