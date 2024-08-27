from pydantic import BaseModel
from typing import List, Tuple

SEGMENTATION_OUTPUT_KEY = "segmentation"

# class labels for map and points legend areas
SEGMENT_MAP_CLASS = "map"
SEGMENT_POINT_LEGEND_CLASS = "legend_points_lines"
SEGMENT_POLYGON_LEGEND_CLASS = "legend_polygons"


class SegmentationResult(BaseModel):
    """
    Class for storing a segmentation result
    """

    poly_bounds: List[
        Tuple[float, float]
    ]  # segmentation polygon, list of xy co-ords (in pixel units)
    area: float  # segmentation area
    bbox: List[float]  # bounding box
    class_label: str  # predicted segmentation class label
    confidence: float  # prediction score
    id_model: str  # model ID (model_id not used due to pydantic reserved word)
    text: str | None = ""  # text associated with this segment (optional)


class MapSegmentation(BaseModel):
    """Class for storing map segmentation results"""

    doc_id: str  # document ID
    segments: List[SegmentationResult]  # list of segmentation results
