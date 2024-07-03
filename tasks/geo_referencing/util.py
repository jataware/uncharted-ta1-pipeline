from tasks.text_extraction.entities import Point
from tasks.common.task import TaskInput
from tasks.geo_referencing.entities import DocGeoFence, GEOFENCE_OUTPUT_KEY, Coordinate
from tasks.metadata_extraction.entities import MetadataExtraction
from util.coordinate import absolute_minmax

from typing import List, Tuple, Dict


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
            absolute_minmax(input.get_request_info("lon_minmax", [0, 180])),
            absolute_minmax(input.get_request_info("lat_minmax", [0, 90])),
            True,
        )

    # when parsing, only the absolute range matters as coordinates may or may not have the negative sign
    return (
        absolute_minmax(geofence.geofence.lon_minmax),
        absolute_minmax(geofence.geofence.lat_minmax),
        geofence.geofence.defaulted,
    )


def is_in_range(value: float, range_minmax: List[float]) -> bool:
    # range will be [min, max] however min & max may or may not be in absolute terms
    # value could be absolute or not so need to account for both possibilities
    if range_minmax[0] * range_minmax[1] >= 0:
        # range is either fully negative or positive so use absolute
        # adjust range to order them by absolute
        range_minmax_updated = list(map(lambda x: abs(x), range_minmax))
        range_minmax_updated.sort()
        return range_minmax_updated[0] <= abs(value) <= range_minmax_updated[1]

    # range crosses 0 so cant rely on naive check since max may be lower than min (ex: min_max may be [2, -30])
    range_minmax_updated = [
        min(range_minmax[0], range_minmax[1]),
        max(range_minmax[0], range_minmax[1]),
    ]

    # either the actual value was parsed, or the absolute value was parsed
    # if the absolute value was parsed, need to separately check min <= x <= 0 via absolutes
    return range_minmax_updated[0] <= value <= range_minmax_updated[
        1
    ] or 0 <= value <= abs(range_minmax_updated[1])


def is_nad_83(metadata: MetadataExtraction) -> bool:
    # assume nad27 unless evidence for nad83
    year = 1900
    if metadata.year.isdigit():
        year = int(metadata.year)

    return "83" in metadata.projection or year >= 1986


def get_min_max_count(
    coordinates: Dict[Tuple[float, float], Coordinate], sources: List[str] = []
) -> Tuple[float, float, int]:
    if len(coordinates) == 0:
        return 0, 0, 0

    coords = filter(
        lambda x: x[1].get_source() in sources if len(sources) > 0 else True,
        coordinates.items(),
    )

    values = list(map(lambda x: x[1].get_parsed_degree(), coords))

    return min(values), max(values), len(values)
