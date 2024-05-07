from tasks.text_extraction.entities import Point
from tasks.common.task import TaskInput
from tasks.geo_referencing.entities import DocGeoFence, GEOFENCE_OUTPUT_KEY
from util.coordinate import absolute_minmax

from typing import List, Tuple


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


def get_input_geofence(input: TaskInput) -> Tuple[List[float], List[float], bool]:
    geofence: DocGeoFence = input.parse_data(
        GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
    )

    if geofence is None:
        return (
            input.get_request_info("lon_minmax", [0, 180]),
            input.get_request_info("lat_minmax", [0, 180]),
            True,
        )

    # when parsing, only the absolute range matters as coordinates may or may not have the negative sign
    return (
        absolute_minmax(geofence.geofence.lon_minmax),
        absolute_minmax(geofence.geofence.lat_minmax),
        geofence.geofence.defaulted,
    )
