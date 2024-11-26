from typing import List, Tuple, Dict
import io
import math

from tasks.text_extraction.entities import Point
from tasks.common.task import TaskInput
from tasks.geo_referencing.entities import (
    DocGeoFence,
    GEOFENCE_OUTPUT_KEY,
    Coordinate,
    GeoFenceType,
)
from tasks.metadata_extraction.entities import MetadataExtraction
from tasks.geo_referencing.entities import (
    CoordSource,
    GroundControlPoint as LARAGroundControlPoint,
)

from util.coordinate import absolute_minmax

from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
import rasterio as rio
import rasterio.transform as riot
import rasterio.io as rioi

from pyproj import Transformer

from PIL.Image import Image as PILImage
import logging


def sign(number: float) -> int:
    """
    sign function, returns 1 or -1
    """
    return int(math.copysign(1, number))


def get_distinct_degrees(coords: Dict[Tuple[float, float], Coordinate]) -> int:
    """
    Get the number of unique degree values for extracted coord values
    """
    coords_distinct = set(map(lambda x: x[1].get_parsed_degree(), coords.items()))
    return len(coords_distinct)


def is_coord_from_source(
    coords: Dict[Tuple[float, float], Coordinate], sources: List[CoordSource]
) -> bool:
    """
    Check if any coords were extracted from target CoordSources
    """
    for key, c in coords.items():
        if c.get_source() in sources:
            return True
    return False


def calc_lonlat_slope_signs(lonlat_hemispheres: Tuple[int, int]):
    """
    Calculates the expected slope direction for degrees-to-pixels
    for latitudes and longitudes
    NOTE: Assumes raw degree values extracted from a map are absolute values
    """

    # --- longitudes (x pixel direction; +ve left->right)
    # eastern hemisphere, longitude values are +ve; abs values are increasing left->right on a map
    lon_slope_sign = 1
    if lonlat_hemispheres[0] < 0:
        # western hemisphere, longitude values are -ve; abs values are decreasing left->right on a map
        lon_slope_sign = -1

    # --- latitudes (y pixel direction; +ve top->bottem)
    # northern hemisphere, latitude values are +ve; abs values are decreasing top->bottem on a map
    lat_slope_sign = -1
    if lonlat_hemispheres[1] < 0:
        # southern hemisphere, latitude values are -ve; abs values are increasing top->bottem on a map
        lat_slope_sign = 1

    return (lon_slope_sign, lat_slope_sign)


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
        geofence.geofence.region_type == GeoFenceType.DEFAULT,
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
    if "83" in metadata.datum:
        return True
    for crs in metadata.coordinate_systems:
        if "83" in crs:
            return True

    year = 1900
    if metadata.year.isdigit():
        year = int(metadata.year)

    return "83" in metadata.projection or year >= 1986


def get_min_max_count(
    coordinates: Dict[Tuple[float, float], Coordinate],
    is_negative_hemisphere: bool,
    sources: List[CoordSource] = [],
) -> Tuple[float, float, int]:
    coords = get_points(coordinates, sources).items()
    if len(coords) == 0:
        return 0, 0, 0

    # adjust values to be in the right hemisphere
    multiplier = 1
    if is_negative_hemisphere:
        multiplier = -1

    values = list(map(lambda x: multiplier * abs(x[1].get_parsed_degree()), coords))

    return min(values), max(values), len(values)


def get_points(
    coordinates: Dict[Tuple[float, float], Coordinate], sources: List[CoordSource] = []
) -> Dict[Tuple[float, float], Coordinate]:

    if coordinates is None or len(coordinates) == 0:
        return {}

    coords = list(
        filter(
            lambda x: x[1].get_source() in sources if len(sources) > 0 else True,
            coordinates.items(),
        )
    )

    filtered = {}
    for c in coords:
        filtered[c[0]] = c[1]
    return filtered


def cps_to_transform(
    gcps: List[LARAGroundControlPoint],
    source_crs: str,
    dest_crs: str,
) -> Affine:
    """
    Transforms ground control points from one coordinate reference system (CRS) to another.
    Args:
        gcps (List[LARAGroundControlPoint]): List of ground control points.
        source_crs (str): Source CRS of the ground control points.
        dest_crs (str): Destination CRS to transform the ground control points to.
    Returns:
        Affine: Affine transformation matrix.
    """
    proj = Transformer.from_crs(source_crs, dest_crs, always_xy=True)
    proj_gcps = [
        riot.GroundControlPoint(
            row=gcp.pixel_y,
            col=gcp.pixel_x,
            x=proj.transform(xx=gcp.longitude, yy=gcp.latitude)[0],
            y=proj.transform(xx=gcp.longitude, yy=gcp.latitude)[1],
        )
        for gcp in gcps
    ]
    return riot.from_gcps(proj_gcps)


def project_image(
    image: PILImage,
    geo_transform: Affine,
    dest_crs: str,
) -> io.BytesIO:
    """
    Projects an image from one coordinate reference system (CRS) to another.
    Args:
        image (PILImage): Image to project.
        geo_transform (Affine): Affine transformation matrix.
        dest_crs (str): Destination CRS to project the image to.
    """

    try:
        # convert the PILImage into a raw TIFF image in memory
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="tiff")
        img_byte_arr = img_byte_arr.getvalue()

        # load the tiff image into a rasterio dataset
        with rioi.MemoryFile(img_byte_arr) as input_memfile:
            with input_memfile.open() as input_dataset:
                # create the profile for the projected image
                bounds = riot.array_bounds(
                    input_dataset.height, input_dataset.width, geo_transform
                )
                projected_transform, projected_width, projected_height = (
                    calculate_default_transform(
                        dest_crs,
                        dest_crs,
                        input_dataset.width,
                        input_dataset.height,
                        *tuple(bounds),
                    )
                )
                projected_kwargs = input_dataset.profile.copy()
                projected_kwargs.update(
                    {
                        "driver": "COG",
                        "crs": {"init": dest_crs},
                        "transform": projected_transform,
                        "width": projected_width,
                        "height": projected_height,
                    }
                )
                # reproject the raw data into a new in-memory rasterio dataset
                input_data = input_dataset.read()
                with rioi.MemoryFile() as out_memfile:
                    with out_memfile.open(**projected_kwargs) as projected_dataset:
                        for i in range(input_dataset.count):
                            _ = reproject(
                                source=input_data[i],
                                destination=rio.band(projected_dataset, i + 1),
                                src_transform=geo_transform,
                                src_crs=dest_crs,
                                dst_transform=projected_transform,
                                dst_crs=dest_crs,
                                resampling=Resampling.bilinear,
                                num_threads=8,
                                warp_mem_limit=256,
                            )
                    # write the raw geotiff into a BytesIO object for downstream processing
                    return io.BytesIO(out_memfile.read())
    except Exception as e:
        # Log the exception
        logging.exception(f"An error occurred: {e}", exc_info=True)
        return io.BytesIO()
