import random

from abc import ABC, abstractmethod
from copy import deepcopy

from geopy.geocoders import Nominatim
from shapely.geometry import Polygon

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedCoordinate,
    GeocodedPlace,
    MetadataExtraction,
    GEOCODED_PLACES_OUTPUT_KEY,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.text_extraction.entities import TextExtraction

from typing import List, Tuple


class GeocodingService(ABC):
    @abstractmethod
    def geocode(self, place: GeocodedPlace) -> Tuple[GeocodedPlace, bool]:
        pass


class NominatimGeocoder(GeocodingService):
    _service: Nominatim

    def __init__(self, timeout: int, limit_hits: int = 4) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-cma-challenge-geocoder"  # type: ignore
        )
        self._limit_hits = limit_hits

    def geocode(self, place: GeocodedPlace) -> Tuple[GeocodedPlace, bool]:
        place_copy = deepcopy(place)
        res = self._service.geocode(
            place.place_name,  # type: ignore
            exactly_one=False,  # type: ignore
            limit=self._limit_hits,  # type: ignore
            country_codes="us",  # type: ignore
        )

        if res is None:
            return place_copy, False
        if len(res) == 0:  # type: ignore
            return place_copy, False

        # assume the first hit is the one that matters for now
        if "boundingbox" not in res[0].raw:  # type: ignore
            return place_copy, False

        # bounding box returned is a list of strings [x1, x2, y1, y2]
        # these should not be absolute but rather the actual bounding boxes.
        place_copy.coordinates = []
        for r in res:  # type: ignore
            bb_coords_raw = list(map(lambda x: float(x), r.raw["boundingbox"]))  # type: ignore
            bb_coords = [
                bb_coords_raw[:2],
                bb_coords_raw[2:],
            ]

            # build the coordinate bounding box via 4 points
            place_copy.coordinates.append(
                [
                    GeocodedCoordinate(
                        pixel_x=place.coordinates[0][0].pixel_x,
                        pixel_y=place.coordinates[0][0].pixel_y,
                        geo_x=bb_coords[1][0],
                        geo_y=bb_coords[0][0],
                    ),
                    GeocodedCoordinate(
                        pixel_x=place.coordinates[0][0].pixel_x,
                        pixel_y=place.coordinates[0][0].pixel_y,
                        geo_x=bb_coords[1][1],
                        geo_y=bb_coords[0][0],
                    ),
                    GeocodedCoordinate(
                        pixel_x=place.coordinates[0][0].pixel_x,
                        pixel_y=place.coordinates[0][0].pixel_y,
                        geo_x=bb_coords[1][1],
                        geo_y=bb_coords[0][1],
                    ),
                    GeocodedCoordinate(
                        pixel_x=place.coordinates[0][0].pixel_x,
                        pixel_y=place.coordinates[0][0].pixel_y,
                        geo_x=bb_coords[1][0],
                        geo_y=bb_coords[0][1],
                    ),
                ]
            )
        return place_copy, True


class Geocoder(Task):
    _geocoding_service: GeocodingService

    def __init__(
        self,
        task_id: str,
        geocoding_service: GeocodingService,
        run_bounds: bool = True,
        run_points: bool = True,
    ):
        super().__init__(task_id)
        self._geocoding_service = geocoding_service
        self._run_bounds = run_bounds
        self._run_points = run_points

    def run(self, input: TaskInput) -> TaskResult:
        print(f"running geocoding task with id {self._task_id}")
        to_geocode = self._get_places(input)

        geocoded_output = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        if geocoded_output is None:
            geocoded_output = DocGeocodedPlaces(map_id=input.raster_id, places=[])

        if self._run_bounds:
            geocoded_output.places = geocoded_output.places + self._geocode_bounds(
                input, to_geocode
            )

        if self._run_points:
            geocoded_output.places = geocoded_output.places + self._geocode_points(
                input, to_geocode
            )

        # update the coordinates list
        return self._create_result(input, geocoded_output)

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
        print(f"METADATA COUNTRY: {metadata.country}")
        print(f"METADATA STATE: {metadata.states}")

        places: List[GeocodedPlace] = []
        country = ""
        if metadata.country and not metadata.country == "NULL":
            country = metadata.country
            places.append(
                GeocodedPlace(
                    place_name=metadata.country,
                    place_location_restriction="",
                    place_type="bound",
                    coordinates=[
                        [GeocodedCoordinate(geo_x=0, geo_y=0, pixel_x=0, pixel_y=0)]
                    ],
                )
            )

        for s in metadata.states:
            if s and not s == "NULL":
                places.append(
                    GeocodedPlace(
                        place_name=s,
                        place_location_restriction=metadata.country,
                        place_type="bound",
                        coordinates=[
                            [GeocodedCoordinate(geo_x=0, geo_y=0, pixel_x=0, pixel_y=0)]
                        ],
                    )
                )

        for p in metadata.places:
            places.append(
                GeocodedPlace(
                    place_name=p.text,
                    place_location_restriction=country,
                    place_type="point",
                    coordinates=[[self._map_coordinates(p)]],
                )
            )

        return places

    def _reduce_to_point(self, place: TextExtraction) -> Tuple[float, float]:
        # for now reduce it to the central point
        polygon = Polygon([(b.x, b.y) for b in place.bounds])
        centroid = polygon.centroid
        return (centroid.x, centroid.y)

    def _geocode_bounds(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> List[GeocodedPlace]:
        geocoded_places = []
        for p in places:
            if p.place_type == "bound":
                g, s = self._geocoding_service.geocode(p)
                if s:
                    geocoded_places.append(g)
        return geocoded_places

    def _geocode_points(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> List[GeocodedPlace]:
        geocoded_places = []
        limit = min(10, len(places))
        places_to_geocode = random.sample(places, limit)
        for p in places_to_geocode:
            if p.place_type == "point":
                g, s = self._geocoding_service.geocode(p)
                if s:
                    geocoded_places.append(g)
        return geocoded_places

    def _map_coordinates(self, extraction: TextExtraction) -> GeocodedCoordinate:
        # map from extraction coordinates to geocoded coordinates by using the centroid
        centroid = self._reduce_to_point(extraction)
        return GeocodedCoordinate(
            geo_x=0, geo_y=0, pixel_x=round(centroid[0]), pixel_y=round(centroid[1])
        )
