import json
import csv
import logging
import os

from abc import ABC, abstractmethod
from copy import deepcopy

from geopy.geocoders import Nominatim

from tasks.metadata_extraction.entities import (
    GeocodedCoordinate,
    GeocodedPlace,
    GeocodedResult,
)


from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GeocodingService(ABC):
    """
    Abstract class for geocoding service
    """

    @abstractmethod
    def geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Tuple[GeocodedPlace, bool]:
        pass


class NominatimGeocoder(GeocodingService):
    """
    Geocoding service using Nominatim, via the geopy package
    """

    _service: Nominatim

    def __init__(
        self,
        timeout: int,
        cache_location: str,
        limit_hits: int = 4,
        country_code_filename: str = "",
        geocoded_places_filename: str = "",
    ) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-lara-geocoder"  # type: ignore
        )
        self._limit_hits = limit_hits
        self._cache = self._load_cache(cache_location)
        self._cache_location = cache_location
        self._country_to_code = self._build_country_lookup(country_code_filename)
        self._code_to_country = {v: k for k, v in self._country_to_code.items()}
        self._geocoded_places_reference = self._build_geocoded_places_reference(
            geocoded_places_filename
        )

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

    def _build_geocoded_places_reference(
        self, geocoded_places_filename: str
    ) -> Dict[str, Dict]:
        """
        Load reference JSON data of pre-geocoded common place names
        (Note: this helps with known bugs for Nominatim geo-bounds for Alaska, etc.)
        """
        try:
            with open(geocoded_places_filename, "rb") as f:
                geocoded_places_ref = json.load(f)
            return geocoded_places_ref
        except Exception as e:
            logger.error(
                f"EXCEPTION loading geocoded places reference JSON at {geocoded_places_filename}; skipping. {repr(e)}"
            )
            return {}

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

    def geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Tuple[GeocodedPlace, bool]:
        """
        Perform Nominatim geocoding on a place name
        """
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

    def _get_country_code(self, place: GeocodedPlace) -> Optional[str]:
        country_code = None
        if len(self._country_to_code) > 0:
            country = place.place_location_restriction.lower()
            # could already be a code
            if len(country) == 2:
                return country
            if country in self._country_to_code:
                country_code = self._country_to_code[country]
        return country_code

    def _get_geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Optional[List[Tuple[List[float], str]]]:

        if place.place_name in self._geocoded_places_reference:
            # a reference geocoded result already exists for this place-name
            res = self._geocoded_places_reference[place.place_name]
            display_name = ""
            if not res["featuretype"] == "country":
                display_name = self._get_state(res["display_name"])
            return [(res["boundingbox"], display_name)]

        featuretype = ""
        if place.place_type == "bound" and place.place_location_restriction == "":
            featuretype = "country"
            # force the parent country code = None if this place represents a country
            parent_country_code = None
        else:
            parent_country_code = self._get_country_code(place)

        key = f"{place.place_name}|{self._limit_hits}|{parent_country_code}|{geofence}|{featuretype}"

        # check cache, assuming cache is a structure {"boundingbox": list[list[float]]}
        if key in self._cache:
            cached_value = self._cache[key]
            results: List[Tuple[List[float], str]] = cached_value["boundingbox"]
            return results

        # use full country name when geo-coding at country level (prevents noisy results with Nominatim)
        place_name = place.place_name
        if featuretype == "country":
            place_name = self._code_to_country.get(place_name.lower(), place_name)

        res = self._service.geocode(
            place_name,  # type: ignore
            exactly_one=False,  # type: ignore
            limit=self._limit_hits,  # type: ignore
            country_codes=parent_country_code,  # type: ignore
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
