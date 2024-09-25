import json
import csv
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
    GeocodedResult,
    MetadataExtraction,
    GEOCODED_PLACES_OUTPUT_KEY,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.text_extraction.entities import TextExtraction

from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("geocoder")


class GeocodingService(ABC):
    @abstractmethod
    def geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Tuple[GeocodedPlace, bool]:
        pass


class NominatimGeocoder(GeocodingService):
    _service: Nominatim

    def __init__(
        self,
        timeout: int,
        cache_location: str,
        limit_hits: int = 4,
        country_code_filename: str = "",
    ) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-lara-geocoder"  # type: ignore
        )
        self._limit_hits = limit_hits
        self._cache = self._load_cache(cache_location)
        self._cache_location = cache_location
        self._country_lookup = self._build_country_lookup(country_code_filename)

    def _build_country_lookup(self, country_code_filename: str) -> Dict[str, str]:
        if len(country_code_filename) == 0:
            return {}

        # read the lookup file
        data = []
        with open(country_code_filename, newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        codes = {}
        for r in data[1:]:
            codes[r[0].lower()] = r[1].lower()
        return codes

    def _load_cache(self, cache_location: str) -> Dict[Any, Any]:
        cache = {}

        # assume the cache is just a json file
        try:
            if os.path.isfile(cache_location):
                with open(cache_location, "rb") as f:
                    cache = json.load(f)
            return cache
        except Exception as e:
            logger.error(
                f"EXCEPTION loading geocoder cache at {cache_location}; reverting cache to empty dict"
            )
            logger.error(e)
            return {}

    def _cache_doc(self, key: str, doc: Any):
        self._cache[key] = doc

        with open(self._cache_location, "w") as f:
            json.dump(self._cache, f)

    def _geocode_place(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Tuple[GeocodedPlace, bool]:
        place_copy = deepcopy(place)
        res_raw = self._get_geocode(place_copy, geofence)

        if res_raw is None:
            return place_copy, False
        if len(res_raw) == 0:
            return place_copy, False

        # bounding box returned is a list of strings [x1, x2, y1, y2]
        # these should not be absolute but rather the actual bounding boxes.
        place_copy.results = []
        for res in res_raw:  # type: ignore
            r = res[0]
            bb_coords = [
                r[:2],
                r[2:],
            ]

            # build the coordinate bounding box via 4 points
            place_copy.results.append(
                GeocodedResult(
                    place_region=res[1],
                    coordinates=[
                        GeocodedCoordinate(
                            pixel_x=place.results[0].coordinates[0].pixel_x,
                            pixel_y=place.results[0].coordinates[0].pixel_y,
                            geo_x=bb_coords[1][0],
                            geo_y=bb_coords[0][0],
                        ),
                        GeocodedCoordinate(
                            pixel_x=place.results[0].coordinates[0].pixel_x,
                            pixel_y=place.results[0].coordinates[0].pixel_y,
                            geo_x=bb_coords[1][1],
                            geo_y=bb_coords[0][0],
                        ),
                        GeocodedCoordinate(
                            pixel_x=place.results[0].coordinates[0].pixel_x,
                            pixel_y=place.results[0].coordinates[0].pixel_y,
                            geo_x=bb_coords[1][1],
                            geo_y=bb_coords[0][1],
                        ),
                        GeocodedCoordinate(
                            pixel_x=place.results[0].coordinates[0].pixel_x,
                            pixel_y=place.results[0].coordinates[0].pixel_y,
                            geo_x=bb_coords[1][0],
                            geo_y=bb_coords[0][1],
                        ),
                    ],
                )
            )
        return place_copy, True

    def geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        is_country: bool = False,
    ) -> Tuple[GeocodedPlace, bool]:
        place_geocoded, result = self._geocode_place(place, geofence)

        return place_geocoded, result

    def _get_country_code(self, place: GeocodedPlace) -> str:
        country_code = ""
        if len(self._country_lookup) > 0:
            country = place.place_location_restriction.lower()
            # could already be a code
            if len(country) == 2:
                return country
            if country in self._country_lookup:
                country_code = self._country_lookup[country]
        return country_code

    def _get_geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Optional[List[Tuple[List[float], str]]]:
        # TODO: update key to use country codes once they no longer get fixed to us
        country_code = self._get_country_code(place)
        if country_code == "":
            country_code = None
        featuretype = ""
        if place.place_type == "bound" and place.place_location_restriction == "":
            featuretype = "country"
        key = f"{place.place_name}|{self._limit_hits}|{country_code}|{geofence}|{featuretype}"

        # check cache, assuming cache is a structure {"boundingbox": list[list[float]]}
        if key in self._cache:
            cached_value = self._cache[key]
            results: List[Tuple[List[float], str]] = cached_value["boundingbox"]
            return results

        res = self._service.geocode(
            place.place_name,  # type: ignore
            exactly_one=False,  # type: ignore
            limit=self._limit_hits,  # type: ignore
            country_codes=country_code,  # type: ignore
            viewbox=geofence,
            featuretype=featuretype,
        )

        if res is None:
            return None
        if len(res) == 0:  # type: ignore
            return None

        # assume the first hit is the one that matters for now
        if "boundingbox" not in res[0].raw:  # type: ignore
            return None

        results = [([float(c) for c in p.raw["boundingbox"]], self._get_state(p.raw["display_name"])) for p in res]  # type: ignore

        # add to cache
        self._cache_doc(key, {"boundingbox": results})

        return results

    def _get_state(self, display_name: str) -> str:
        parts = display_name.split(",")
        if len(parts) < 2:
            return ""

        # state is second last part of display name unless that is a zip code, in which case state is the one before
        state = parts[-2].strip()
        if state.isdigit():
            state = display_name.split(",")[-3].strip()
        return state


class Geocoder(Task):
    _geocoding_service: GeocodingService

    def __init__(
        self,
        task_id: str,
        geocoding_service: GeocodingService,
        run_bounds: bool = True,
        run_points: bool = True,
        run_centres: bool = True,
        should_run: Optional[Callable] = None,
    ):
        super().__init__(task_id)
        self._geocoding_service = geocoding_service
        self._run_bounds = run_bounds
        self._run_points = run_points
        self._run_centres = run_centres
        self._should_run = should_run

    def run(self, input: TaskInput) -> TaskResult:
        metadata: MetadataExtraction = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        geocoded_output = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        if geocoded_output is None:
            geocoded_output = DocGeocodedPlaces(map_id=input.raster_id, places=[])

        if self._should_run and not self._should_run(input):
            logging.info("Skipping geocoding task")
            return self._create_result(input, geocoded_output)

        logger.info(f"running geocoding task with id {self._task_id}")
        to_geocode = self._get_places(input)

        new_places = self._geocode_list(input, to_geocode)
        logger.info(f"geocoded {len(new_places)} places")
        if metadata.country.lower() == "us":
            narrow_geofence = self._narrow_geofence(input, new_places)
            logger.info(f"narrowed geofence determined to be '{narrow_geofence}'")
            if narrow_geofence is not None and len(narrow_geofence) > 0:
                logger.info("rerunning geocoding using narrowed geofence")
                new_places = self._geocode_list(
                    input, to_geocode, geofence=narrow_geofence
                )
                logger.info(f"narrowing geofence geocoded {len(new_places)} places")

        geocoded_output.places = geocoded_output.places + new_places

        # update the coordinates list
        return self._create_result(input, geocoded_output)

    def _geocode_list(
        self, input: TaskInput, to_geocode: List[GeocodedPlace], geofence: str = ""
    ) -> List[GeocodedPlace]:

        geobounds = None
        if geofence is not None and len(geofence) > 0:
            # geocode the geofence to get the bounds
            geobounds_raw = self._geocode_bounds(
                input,
                [
                    GeocodedPlace(
                        place_name=geofence,
                        place_location_restriction=to_geocode[
                            0
                        ].place_location_restriction,
                        place_type="bound",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[
                                    GeocodedCoordinate(
                                        geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
            if len(geobounds_raw) > 0:
                geobounds_raw = geobounds_raw[0]
                lons = list(
                    map(lambda x: x.geo_x, geobounds_raw.results[0].coordinates)
                )
                lats = list(
                    map(lambda x: x.geo_y, geobounds_raw.results[0].coordinates)
                )
                geobounds = ((min(lats), min(lons)), (max(lats), max(lons)))

        new_places = []
        new_places = new_places + self._geocode_bounds(input, to_geocode)
        new_places = new_places + self._geocode_points(input, to_geocode, geobounds)
        new_places = new_places + self._geocode_centres(input, to_geocode, geobounds)

        return new_places

    def _narrow_geofence(
        self, input: TaskInput, places: List[GeocodedPlace]
    ) -> Optional[str]:
        # check if any geocoded place is restricted to a single state
        state = ""
        for p in places:
            state = ""
            for r in p.results:
                if state == "":
                    # initialize the state
                    state = r.place_region
                elif not state == r.place_region:
                    # a new non empty state value means the geofence cannot be narrowed
                    state = ""
                    break
        return state

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
        if metadata.country and not metadata.country == "NULL":
            country = metadata.country
        if self._run_bounds:
            if len(country) > 0:
                places.append(
                    GeocodedPlace(
                        place_name=metadata.country,
                        place_location_restriction="",
                        place_type="bound",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[
                                    GeocodedCoordinate(
                                        geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                    )
                                ],
                            )
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
                            results=[
                                GeocodedResult(
                                    place_region="",
                                    coordinates=[
                                        GeocodedCoordinate(
                                            geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                        )
                                    ],
                                )
                            ],
                        )
                    )

        if self._run_points:
            for p in metadata.places:
                if not isinstance(p, TextExtraction):
                    logger.error("place is not a text extraction")
                    continue
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="point",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[self._map_coordinates(p)],
                            )
                        ],
                    )
                )

        if self._run_centres:
            for p in metadata.population_centres:
                if not isinstance(p, TextExtraction):
                    logger.error("population centre is not a text extraction")
                    continue
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="population",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[self._map_coordinates(p)],
                            )
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
        self,
        input: TaskInput,
        places: List[GeocodedPlace],
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> List[GeocodedPlace]:
        geocoded_places = []
        for p in places:
            if p.place_type == "population":
                g, s = self._geocoding_service.geocode(p, geofence=geofence)
                if s:
                    geocoded_places.append(g)
        return geocoded_places

    def _geocode_points(
        self,
        input: TaskInput,
        places: List[GeocodedPlace],
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> List[GeocodedPlace]:
        geocoded_places = []

        points_only = list(filter(lambda x: x.place_type == "point", places))
        limit = min(200, len(points_only))
        places_to_geocode = random.sample(points_only, limit)
        for p in places_to_geocode:
            g, s = self._geocoding_service.geocode(p, geofence=geofence)
            if s:
                geocoded_places.append(g)
        return geocoded_places

    def _map_coordinates(self, extraction: TextExtraction) -> GeocodedCoordinate:
        # map from extraction coordinates to geocoded coordinates by using the centroid
        centroid = self._reduce_to_point(extraction)
        return GeocodedCoordinate(
            geo_x=0, geo_y=0, pixel_x=round(centroid[0]), pixel_y=round(centroid[1])
        )
