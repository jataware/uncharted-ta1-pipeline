from tasks.text_extraction.entities import Point

from typing import List


def ocr_to_coordinates(bounds: List[Point]) -> List[List[int]]:
    mapped = []
    for v in bounds:
        mapped.append([v.x, v.y])
    return mapped


def get_bounds_bounding_box(bounds: List[Point]) -> List[Point]:
    # reduce a polygon to a bounding box
    x_range = list(map(lambda x: x.x, bounds))
    y_range = list(map(lambda x: x.y, bounds))
    min_x, max_x = min(x_range), max(x_range)
    min_y, max_y = min(y_range), max(y_range)

    return [
        Point(x=min_x, y=min_y),
        Point(x=max_x, y=min_y),
        Point(x=max_x, y=max_y),
        Point(x=min_x, y=max_y),
    ]
