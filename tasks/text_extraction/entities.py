from typing import List
from PIL.Image import Image as PILImage
from typing import Tuple
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


class TextExtraction(BaseModel):
    text: str
    confidence: float
    bounds: List[Point]


class DocTextExtraction(BaseModel):
    """ """

    doc_id: str
    extractions: List[TextExtraction] = []


class Tile:
    """
    Tile representing a partial image
    """

    image: PILImage
    coordinates = (0, 0)

    def __init__(self, image: PILImage, coordinates: Tuple[int, int]):
        self.image = image
        self.coordinates = coordinates
