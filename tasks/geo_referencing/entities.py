from tasks.text_extraction.entities import Point

from typing import List, Tuple
from pydantic import BaseModel, ConfigDict


GEOFENCE_OUTPUT_KEY = "geofence_output"


class GeoFence(BaseModel):
    lat_minmax: List[float]
    lon_minmax: List[float]
    defaulted: bool


class DocGeoFence(BaseModel):
    map_id: str
    geofence: GeoFence


class Coordinate:
    _type: str = ""
    _text: str = ""
    _parsed_degree: float = -1
    _is_lat: bool = False
    _bounds: List[Point] = []
    _pixel_alignment: Tuple[float, float] = (0, 0)
    _confidence: float = 0

    def __init__(
        self,
        type: str,
        text: str,
        parsed_degree: float,
        is_lat: bool = False,
        bounds: List[Point] = [],
        pixel_alignment=None,
        x_ranges: Tuple[float, float] = (0, 1),
        font_height: float = 0.0,
        confidence: float = 0,
    ):
        self._type = type
        self._text = text
        self._bounds = [] if bounds == [] else bounds
        self._parsed_degree = parsed_degree
        self._is_lat = is_lat
        self._confidence = confidence

        if pixel_alignment:
            self._pixel_alignment = pixel_alignment
        elif len(bounds) > 0:
            self._pixel_alignment = self._calculate_pixel_alignment(
                bounds, x_ranges, font_height
            )

    def get_pixel_alignment(self) -> Tuple[float, float]:
        return self._pixel_alignment

    def get_type(self) -> str:
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

    def get_constant_dimension(self) -> float:
        # lat coordinates should be aligned on y axis
        if self._is_lat:
            return self._pixel_alignment[1]
        return self._pixel_alignment[0]

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
