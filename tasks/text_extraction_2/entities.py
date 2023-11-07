from typing import List
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


class TextExtraction(BaseModel):
    text: str
    confidence: float
    bounds: List[Point]


class DocTextExtraction(BaseModel):
    doc_id: str
    extractions: List[TextExtraction] = []
