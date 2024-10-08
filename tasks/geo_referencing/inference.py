import logging
import statistics
import uuid

from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.geo_referencing.entities import (
    Coordinate,
    SOURCE_INFERENCE,
    MapROI,
    ROI_MAP_OUTPUT_KEY,
)
from tasks.geo_referencing.util import ocr_to_coordinates

from typing import Tuple, Dict, List

logger = logging.getLogger("inference_extractor")


class InferenceCoordinate:
    pixel: float
    degree: float


class InferenceCoordinateExtractor(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _should_run(self, input: CoordinateInput) -> bool:
        # do not infer any coordinates if the clue point is provided
        clue_point = input.input.get_request_info("clue_point")
        if clue_point is not None:
            return False

        # should only run if there are sufficient coordinates in one direction
        # and insufficient coordinates in the other direction
        lats = input.input.get_data("lats", [])
        lons = input.input.get_data("lons", [])

        lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lats.items()))
        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lons.items()))

        min_coord, max_coord = min(len(lats_distinct), len(lons_distinct)), max(
            len(lats_distinct), len(lons_distinct)
        )
        return min_coord == 1 and max_coord >= 2

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        # get the coordinates and roi or assume whole image is a map
        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")
        roi_xy = []
        if ROI_MAP_OUTPUT_KEY in input.input.data:
            # get map ROI bounds (without inner/outer buffering)
            map_roi = MapROI.model_validate(input.input.data[ROI_MAP_OUTPUT_KEY])
            roi_xy = map_roi.map_bounds

        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lon_pts.items()))
        infer_lon = len(lons_distinct) < 2
        if infer_lon:
            logger.info("inferring longitude coordinate from latitudes")
            lon = list(lon_pts.items())[0]
            existing_lon = InferenceCoordinate()
            existing_lon.pixel = lon[1].get_pixel_alignment()[0]
            existing_lon.degree = lon[1].get_parsed_degree()
            coordinates = list(map(lambda x: x[1].get_parsed_degree(), lat_pts.items()))
            pixels = list(map(lambda x: x[1].get_pixel_alignment()[1], lat_pts.items()))

            new_lon = self._infer_coordinates(
                coordinates,
                pixels,
                list(map(lambda x: x[0], roi_xy)),
                existing_lon,
                False,
            )
            coord_update = lon_pts
            new_coord = Coordinate(
                "lon keypoint",
                "",
                new_lon.degree,
                SOURCE_INFERENCE,
                False,
                pixel_alignment=(new_lon.pixel, lon[1].get_pixel_alignment()[1]),
                confidence=0.5,
            )
        else:
            logger.info("inferring latitude coordinate from longitudes")
            lat = list(lat_pts.items())[0]
            existing_lat = InferenceCoordinate()
            existing_lat.pixel = lat[1].get_pixel_alignment()[1]
            existing_lat.degree = lat[1].get_parsed_degree()
            coordinates = list(map(lambda x: x[1].get_parsed_degree(), lon_pts.items()))
            pixels = list(map(lambda x: x[1].get_pixel_alignment()[0], lon_pts.items()))

            new_lat = self._infer_coordinates(
                coordinates,
                pixels,
                list(map(lambda x: x[1], roi_xy)),
                existing_lat,
                True,
            )
            coord_update = lat_pts
            new_coord = Coordinate(
                "lat keypoint",
                "",
                new_lat.degree,
                SOURCE_INFERENCE,
                True,
                pixel_alignment=(lat[1].get_pixel_alignment()[0], new_lat.pixel),
                confidence=0.5,
            )

        coord_update[new_coord.to_deg_result()[0]] = new_coord
        self._add_param(
            input.input,
            str(uuid.uuid4()),
            f"coordinate-{new_coord.get_type()}",
            {
                "bounds": ocr_to_coordinates(new_coord.get_bounds()),
                "text": new_coord.get_text(),
                "parsed": new_coord.get_parsed_degree(),
                "type": "latitude" if new_coord.is_lat() else "longitude",
                "pixel_alignment": new_coord.get_pixel_alignment(),
                "confidence": new_coord.get_confidence(),
            },
            "extracted coordinate",
        )

        return (lon_pts, lat_pts)

    def _get_range(self, range: List[float]) -> float:
        return max(range) - min(range)

    def _get_pixel_degree_ratio(
        self, degrees_valid: List[float], pixels_valid: List[float]
    ) -> float:
        pixel_range = self._get_range(pixels_valid)
        degrees_range = self._get_range(degrees_valid)

        return pixel_range / degrees_range

    def _infer_coordinates(
        self,
        degrees_valid: List[float],
        pixels_valid: List[float],
        pixels_cross: List[float],
        known_coordinate: InferenceCoordinate,
        is_lat: bool,
    ) -> InferenceCoordinate:
        # TODO: figure out the multiplier bit using the geofence
        # get the pixels to degree ratio using the valid direction
        pixels_to_degrees = abs(
            self._get_pixel_degree_ratio(degrees_valid, pixels_valid)
        )
        pixel_range_cross = self._get_range(pixels_cross)

        pixels_cross_min = min(pixels_cross)
        pixels_cross_max = max(pixels_cross)
        pixels_cross_avg = statistics.mean(pixels_cross)

        # set the inference pixel to 80% of the range in the opposite half of the known coordinate
        # multiplier needs to be adjusted based on relative position of inferred point assuming north is up
        #   longitude - right should always be greater than left so increasing pixel should increase degree
        #   latitude - top should always be greater than bottom
        new_coordinate = InferenceCoordinate()
        if known_coordinate.pixel < pixels_cross_avg:
            new_coordinate.pixel = pixels_cross_max - (0.2 * pixel_range_cross)
            multiplier = 1
        else:
            new_coordinate.pixel = pixels_cross_min + (0.2 * pixel_range_cross)
            multiplier = -1

        # latitude degree to pixel relationship is inverse of longitude
        if is_lat:
            multiplier = multiplier * -1

        infer_pixel_range = abs(new_coordinate.pixel - known_coordinate.pixel)

        new_coordinate.degree = known_coordinate.degree + (
            (infer_pixel_range / pixels_to_degrees) * multiplier
        )

        return new_coordinate

    def _derive_hemisphere_multiplier(self, minmax: List[float]) -> int:
        # use the hemisphere of the majority of the range determined by using the sign of the midpoint
        return 1 if ((max(minmax) - min(minmax)) / 2) > 0 else -1
