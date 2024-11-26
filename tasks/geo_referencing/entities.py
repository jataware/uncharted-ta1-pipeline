from tasks.text_extraction.entities import Point
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel

# keys for the pipeline result dictionary
GEOFENCE_OUTPUT_KEY = "geofence_output"
CORNER_POINTS_OUTPUT_KEY = "corner_points"
CRS_OUTPUT_KEY = "crs"
QUERY_POINTS_OUTPUT_KEY = "query_pts"
PROJECTED_MAP_OUTPUT_KEY = "projected_map"
SUMMARY_OUTPUT_KEY = "summary_output"
SCORING_OUTPUT_KEY = "scoring_output"
LEVERS_OUTPUT_KEY = "levers_output"
RMSE_OUTPUT_KEY = "rmse"
ERROR_SCALE_OUTPUT_KEY = "error_scale"
KEYPOINTS_OUTPUT_KEY = "keypoints"
GEOREFERENCING_OUTPUT_KEY = "georeferencing"
ROI_MAP_OUTPUT_KEY = "map_roi"
MAP_SCALE_OUTPUT_KEY = "map_scale"


class MapROI(BaseModel):
    map_bounds: List[Tuple[float, float]]
    buffer_outer: List[Tuple[float, float]]
    buffer_inner: List[Tuple[float, float]]


class GeoFenceType(str, Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"
    GEO_BOUNDS = "geo_bounds"
    CLUE = "clue"
    DEFAULT = "default"


class GeoFence(BaseModel):
    lat_minmax: List[float]
    lon_minmax: List[float]
    region_type: GeoFenceType
    lonlat_hemispheres: Tuple[int, int]


class DocGeoFence(BaseModel):
    map_id: str
    geofence: GeoFence


class MapScale(BaseModel):
    scale_raw: str
    dpi: int
    scale_pixels: float
    km_per_pixel: float
    degrees_per_pixel: float


class GroundControlPoint(BaseModel):
    id: str
    pixel_x: float
    pixel_y: float
    latitude: float
    longitude: float
    confidence: float


class GeoreferenceResult(BaseModel):
    map_id: str
    crs: str
    gcps: List[GroundControlPoint]
    provenance: str
    confidence: float


class CoordType(str, Enum):
    KEYPOINT = "keypoint"  # generic geo-coord for x or y direction (lat, lon, UTM, etc)
    CORNER = "corner"  # map corner point (eg lat/lon pair together)
    GEOCODED_POINT = "geocoded_point"  # geocoded place (with lat/lon pair together)
    DERIVED_KEYPOINT = "derived_keypoint"  # estimated keypoint based on other keypoints, map scale, etc.


class CoordSource(str, Enum):
    LAT_LON = "lat_lon_parser"
    STATE_PLANE = "state_plane_parser"
    UTM = "utm_parser"
    GEOCODER = "geocoder"
    INFERENCE = "inference"
    ANCHOR = "anchor"


class CoordStatus(str, Enum):
    OK = "ok"
    OUTSIDE_MAP_ROI = "outside_map_roi"
    OUTSIDE_GEOFENCE = "outside_geofence"
    OUTLIER = "outlier"


class Coordinate:
    """
    Class for storing an extracted or derived geo-coordinate
    """

    _type: CoordType  # coordinate type
    _text: str  # raw OCR text associated with the extraction
    _parsed_degree: float  # parsed coordinate degree
    _source: CoordSource  # extractor source (provinence)
    _is_lat: bool  # lat or lon?
    _bounds: List[Point]  # pixel bounds of extracted coordinate
    _pixel_alignment: Tuple[float, float]  # centroid pixel
    _confidence: float  # extraction confidence
    _status: CoordStatus  # included/excluded status of this extraction

    def __init__(
        self,
        type: CoordType,
        text: str,
        parsed_degree: float,
        source: CoordSource,
        is_lat: bool,
        bounds: List[Point] = [],
        pixel_alignment: Optional[Tuple[float, float]] = None,
        x_ranges: Tuple[float, float] = (0, 1),
        font_height: float = 0.0,
        confidence: float = 0.0,
        status: CoordStatus = CoordStatus.OK,
    ):
        self._type = type
        self._text = text
        self._source = source
        self._bounds = [] if bounds == [] else bounds
        self._parsed_degree = parsed_degree
        self._is_lat = is_lat
        self._confidence = confidence
        self._status = status

        if pixel_alignment:
            self._pixel_alignment = pixel_alignment
        elif len(bounds) > 0:
            self._pixel_alignment = self._calculate_pixel_alignment(
                bounds, x_ranges, font_height
            )

    def get_pixel_alignment(self) -> Tuple[float, float]:
        """
        Get pixel center xy for this co-ordinate extraction
        """
        return self._pixel_alignment

    def get_type(self) -> CoordType:
        return self._type

    def get_text(self) -> str:
        return self._text

    def get_bounds(self) -> List[Point]:
        return self._bounds

    def get_parsed_degree(self) -> float:
        return self._parsed_degree

    def get_confidence(self) -> float:
        return self._confidence

    def is_lat(self) -> bool:
        return self._is_lat

    def get_source(self) -> CoordSource:
        return self._source

    def get_constant_dimension(self) -> float:
        # lat coordinates should be aligned on y axis
        if self._is_lat:
            return self._pixel_alignment[1]
        return self._pixel_alignment[0]

    def set_pixel_alignment(self, pixel_alignment: Tuple[float, float]):
        self._pixel_alignment = pixel_alignment

    def to_deg_result(self) -> Tuple[Tuple[float, float], float]:
        if self._is_lat:
            return (
                self._parsed_degree,
                self._pixel_alignment[1],
            ), self._pixel_alignment[0]
        return (self._parsed_degree, self._pixel_alignment[0]), self._pixel_alignment[1]

    def _calculate_pixel_alignment(
        self,
        bounds: List[Point],
        x_ranges: Tuple[float, float],
        font_height: float = 0.0,
    ) -> Tuple[float, float]:
        x_pixel = self._get_center_x(bounds, x_ranges)
        y_pixel = self._get_center_y(bounds) + font_height / 2

        return (x_pixel, y_pixel)

    def _get_center_y(self, bounds: List[Point]) -> float:
        min_y = bounds[0].y
        max_y = bounds[3].y
        return (min_y + max_y) / 2.0

    def _get_center_x(
        self, bounds: List[Point], x_ranges: Tuple[float, float]
    ) -> float:
        min_x = bounds[0].x
        max_x = bounds[2].x
        if x_ranges[0] > 0.0 or x_ranges[1] < 1.0:
            x_span = max_x - min_x
            min_x += x_span * x_ranges[0]
            max_x -= x_span * (1.0 - x_ranges[1])
        return (min_x + max_x) / 2.0
