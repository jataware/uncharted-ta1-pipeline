
from pydantic import BaseModel
from typing import List, Tuple


class SegmentationResult(BaseModel):
    '''
    Class for storing a segmentation result
    '''

    poly_bounds: List[Tuple]     #segmentation polygon, list of xy co-ords (in pixel units)
    area: float             # segmentation area
    bbox: List[float]       # bounding box
    class_label: str        #predicted segmentation class label
    confidence: float       # prediction score
    model_id: str           # model ID

        