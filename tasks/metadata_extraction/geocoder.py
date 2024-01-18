from abc import ABC, abstractmethod
from copy import deepcopy

from geopy.geocoders import Nominatim

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedCoordinate,
    GeocodedPlace,
    MetadataExtraction,
    GEOCODED_PLACES_OUTPUT_KEY,
    METADATA_EXTRACTION_OUTPUT_KEY,
)

from typing import List


class GeocodingService(ABC):
    @abstractmethod
    def geocode(self, place: GeocodedPlace) -> GeocodedPlace:
        pass


class NominatimGeocoder(GeocodingService):
    _service: Nominatim

    def __init__(self, timeout: int) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-cma-challenge-geocoder"  # type: ignore
        )

    def geocode(self, place: GeocodedPlace) -> GeocodedPlace:
        place_copy = deepcopy(place)
        res = self._service.geocode(
            place.place_name,  # type: ignore
            exactly_one=False,  # type: ignore
            limit=4,  # type: ignore
            country_codes="us",  # type: ignore
        )

        if res is None:
            return place_copy
        if len(res) == 0:  # type: ignore
            return place_copy

        # assume the first hit is the one that matters for now
        if "boundingbox" not in res[0].raw:  # type: ignore
            return place_copy

        # bounding box returned is a list of strings [x1, x2, y1, y2]
        # these should not be absolute but rather the actual bounding boxes.
        bb_coords_raw = list(map(lambda x: float(x), res[0].raw["boundingbox"]))  # type: ignore
        bb_coords = [
            bb_coords_raw[:2],
            bb_coords_raw[2:],
        ]

        # build the coordinate bounding box via 4 points
        place_copy.coordinates = [
            GeocodedCoordinate(
                pixel_x=0, pixel_y=0, geo_x=bb_coords[1][0], geo_y=bb_coords[0][0]
            ),
            GeocodedCoordinate(
                pixel_x=0, pixel_y=0, geo_x=bb_coords[1][1], geo_y=bb_coords[0][0]
            ),
            GeocodedCoordinate(
                pixel_x=0, pixel_y=0, geo_x=bb_coords[1][1], geo_y=bb_coords[0][1]
            ),
            GeocodedCoordinate(
                pixel_x=0, pixel_y=0, geo_x=bb_coords[1][0], geo_y=bb_coords[0][1]
            ),
        ]
        return place_copy


class Geocoder(Task):
    _geocoding_service: GeocodingService

    def __init__(self, task_id: str, geocoding_service: GeocodingService):
        super().__init__(task_id)
        self._geocoding_service = geocoding_service

    def run(self, input: TaskInput) -> TaskResult:
        to_geocode = self._get_places(input)

        geocoded = self._geocode(input, to_geocode)

        # update the coordinates list
        return self._create_result(input, geocoded)

    def _create_result(
        self,
        input: TaskInput,
        geocoded: DocGeocodedPlaces,
    ) -> TaskResult:
        result = super()._create_result(input)
        result.add_output(GEOCODED_PLACES_OUTPUT_KEY, geocoded.model_dump())

        return result

    def _get_places(self, input: TaskInput) -> List[GeocodedPlace]:
        # extract places from extracted metadata
        metadata: MetadataExtraction = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        if metadata is None:
            return []

        places: List[GeocodedPlace] = []
        places.append(
            GeocodedPlace(
                place_name=metadata.country,
                place_location_restriction="",
                place_type="bound",
                coordinates=[],
            )
        )

        for s in metadata.states:
            places.append(
                GeocodedPlace(
                    place_name=s,
                    place_location_restriction=metadata.country,
                    place_type="bound",
                    coordinates=[],
                )
            )

        return places

    def _geocode(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> DocGeocodedPlaces:
        geocoded_output = DocGeocodedPlaces(map_id=input.raster_id, places=[])
        for p in places:
            geocoded_output.places.append(self._geocoding_service.geocode(p))
        return geocoded_output
