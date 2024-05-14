import json
import logging
import os
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

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("geocoder")


class GeocodingService(ABC):
    @abstractmethod
    def geocode(self, place: GeocodedPlace) -> Tuple[GeocodedPlace, bool]:
        pass


class NominatimGeocoder(GeocodingService):
    _service: Nominatim

    def __init__(self, timeout: int, cache_location: str, limit_hits: int = 4) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-lara-geocoder"  # type: ignore
        )
        self._limit_hits = limit_hits
        self._cache = self._load_cache(cache_location)
        self._cache_location = cache_location

    def _load_cache(self, cache_location: str) -> Dict[Any, Any]:
        cache = {}

        # assume the cache is just a json file
        if os.path.isfile(cache_location):
            with open(cache_location, "rb") as f:
                cache = json.load(f)
        return cache

    def _cache_doc(self, key: str, doc: Any):
        self._cache[key] = doc

        with open(self._cache_location, "w") as f:
            json.dump(self._cache, f)

    def _geocode_place(self, place: GeocodedPlace) -> Tuple[GeocodedPlace, bool]:
        place_copy = deepcopy(place)
        res = self._get_geocode(place_copy)
        print(place)

        if res is None:
            return place_copy, False
        if len(res) == 0:
            return place_copy, False

        # bounding box returned is a list of strings [x1, x2, y1, y2]
        # these should not be absolute but rather the actual bounding boxes.
        place_copy.coordinates = []
        for r in res:  # type: ignore
            bb_coords = [
                r[:2],
                r[2:],
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

    def geocode(self, place: GeocodedPlace) -> Tuple[GeocodedPlace, bool]:
        place_geocoded, result = self._geocode_place(place)

        return place_geocoded, result

    def _get_geocode(self, place: GeocodedPlace) -> Optional[List[List[float]]]:
        # TODO: update key to use country codes once they no longer get fixed to us
        key = f"{place.place_name}|{self._limit_hits}|us"

        # check cache, assuming cache is a structure {"boundingbox": list[list[float]]}
        if key in self._cache:
            cached_value = self._cache[key]
            results: List[List[float]] = cached_value["boundingbox"]
            return results

        res = self._service.geocode(
            place.place_name,  # type: ignore
            exactly_one=False,  # type: ignore
            limit=self._limit_hits,  # type: ignore
            country_codes="us",  # type: ignore
        )

        if res is None:
            return None
        if len(res) == 0:  # type: ignore
            return None

        # assume the first hit is the one that matters for now
        if "boundingbox" not in res[0].raw:  # type: ignore
            return None

        results = [[float(c) for c in p.raw["boundingbox"]] for p in res]  # type: ignore

        # add to cache
        self._cache_doc(key, {"boundingbox": results})

        return results


class Geocoder(Task):
    _geocoding_service: GeocodingService

    def __init__(
        self,
        task_id: str,
        geocoding_service: GeocodingService,
        run_bounds: bool = True,
        run_points: bool = True,
        run_centres: bool = True,
    ):
        super().__init__(task_id)
        self._geocoding_service = geocoding_service
        self._run_bounds = run_bounds
        self._run_points = run_points
        self._run_centres = run_centres

    def run(self, input: TaskInput) -> TaskResult:
        logger.info(f"running geocoding task with id {self._task_id}")
        to_geocode = self._get_places(input)

        geocoded_output = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        if geocoded_output is None:
            geocoded_output = DocGeocodedPlaces(map_id=input.raster_id, places=[])

        geocoded_output.places = geocoded_output.places + self._geocode_bounds(
            input, to_geocode
        )
        geocoded_output.places = geocoded_output.places + self._geocode_points(
            input, to_geocode
        )
        geocoded_output.places = geocoded_output.places + self._geocode_centres(
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

        places: List[GeocodedPlace] = []
        country = ""
        if self._run_bounds:
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
                                [
                                    GeocodedCoordinate(
                                        geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                    )
                                ]
                            ],
                        )
                    )

        if self._run_points:
            for p in metadata.places:
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="point",
                        coordinates=[[self._map_coordinates(p)]],
                    )
                )

        if self._run_centres:
            for p in metadata.population_centres:
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="population",
                        coordinates=[
                            [GeocodedCoordinate(geo_x=0, geo_y=0, pixel_x=0, pixel_y=0)]
                        ],
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

    def _geocode_centres(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> List[GeocodedPlace]:
        geocoded_places = []
        for p in places:
            if p.place_type == "population":
                g, s = self._geocoding_service.geocode(p)
                if s:
                    geocoded_places.append(g)
        return geocoded_places

    def _geocode_points(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> List[GeocodedPlace]:
        geocoded_places = []

        points_only = list(filter(lambda x: x.place_type == "point", places))
        limit = min(10, len(points_only))
        places_to_geocode = random.sample(points_only, limit)
        for p in places_to_geocode:
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
