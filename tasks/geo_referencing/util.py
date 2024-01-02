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


def absolute_minmax(minmax: list[float]) -> list[float]:
    minmax_abs = minmax.copy()
    # if the min max crosses 0, need to have it span from 0 to the furthest value
    if minmax_abs[0] < 0 and minmax_abs[1] >= 0:
        minmax_abs = [0, max(abs(minmax_abs[0]), abs(minmax_abs[1]))]
    else:
        minmax_abs = [abs(minmax_abs[0]), abs(minmax_abs[1])]
        minmax_abs = [min(minmax_abs), max(minmax_abs)]
    return minmax_abs
