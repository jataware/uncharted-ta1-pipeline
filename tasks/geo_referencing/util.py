from tasks.text_extraction.entities import Point

from typing import List


def ocr_to_coordinates(bounds: List[Point]) -> List[List[int]]:
    mapped = []
    for v in bounds:
        mapped.append([v.x, v.y])
    return mapped
