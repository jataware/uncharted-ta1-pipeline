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
    GeoPlaceType,
    GeoFeatureType,
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

    @abstractmethod
    def normalize_country(self, country: str) -> Tuple:
        pass

    @abstractmethod
    def normalize_state(self, state: str) -> Tuple:
        pass

    @abstractmethod
    def normalize_county(self, county: str, state: str) -> str:
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
        state_code_filename: str = "",
        geocoded_places_filename: str = "",
    ) -> None:
        self._service = Nominatim(
            timeout=timeout, user_agent="uncharted-lara-geocoder"  # type: ignore
        )
        self._limit_hits = limit_hits
        self._cache = self._load_cache(cache_location)
        self._cache_location = cache_location
        self._country_to_code = self._build_placecode_lookup(country_code_filename)
        self._code_to_country = {v: k for k, v in self._country_to_code.items()}
        self._code_to_state = {
            v: k for k, v in self._build_placecode_lookup(state_code_filename).items()
        }
        self._geocoded_places_reference = self._build_geocoded_places_reference(
            geocoded_places_filename
        )

    def _build_placecode_lookup(self, place_code_filename: str) -> Dict[str, str]:
        if len(place_code_filename) == 0:
            return {}

        # read the lookup file
        data = []
        with open(place_code_filename, newline="") as f:
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

    def _get_geocode(
        self,
        place: GeocodedPlace,
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Optional[List[Tuple[List[float], str]]]:
        """
        Get the geo-coding result for a given place
        (either via Nominatim query or from LARA's geocoding cache)
        """

        if place.place_name.lower() in self._geocoded_places_reference:
            # a reference geocoded result already exists for this place-name
            res = self._geocoded_places_reference[place.place_name.lower()]
            display_name = ""
            if not res["featuretype"] == "country":
                display_name = self._get_state(res["display_name"])
            return [(res["boundingbox"], display_name)]

        place_name = place.place_name
        featuretype = ""
        parent_country_code = None
        if place.feature_type == GeoFeatureType.COUNTRY:
            featuretype = "country"
        else:
            parent_country_code = place.parent_country
            if place.feature_type == GeoFeatureType.STATE:
                featuretype = "state"

        key = f"{place_name}|{self._limit_hits}|{parent_country_code}|{geofence}|{featuretype}"

        # check cache, assuming cache is a structure {"boundingbox": list[list[float]]}
        if key in self._cache:
            cached_value = self._cache[key]
            results: List[Tuple[List[float], str]] = cached_value["boundingbox"]
            return results

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

    def normalize_country(self, country: str) -> Tuple:
        """
        Normalize the country name for a given place
        returns a Tuple of country name, country code
        """
        country_code = None
        country_name = None
        country = country.lower()
        # could already be a code
        if country in self._code_to_country:
            country_name = self._code_to_country[country]
            country_code = country
        elif country in self._country_to_code:
            country_name = self._country_to_code[country]
            country_code = country
        else:
            country_name = country
        return country_name, country_code

    def normalize_state(self, state: str) -> Tuple:
        """
        Parse and normalize the state code and name for a given place
        """
        state_code = None
        state_name = None
        state = state.lower()
        # could already be a code
        if state in self._code_to_state:
            state_code = state
            state_name = self._code_to_state[state]
        else:
            # try to parse state code from 'place_name', if of the form (US-MI)
            countrystate = state.split("-")
            if len(countrystate) == 2 and len(countrystate[1]) == 2:
                state_code = countrystate[1]
                state_name = self._code_to_state.get(state_code, state_code)
            else:
                state_name = state
        return state_name, state_code

    def normalize_county(self, county: str, state: str) -> str:
        """
        Normalize county name
        """
        if not county:
            return county
        county = county.lower()
        if not county.endswith("county"):
            county = county.strip() + " county"
        if state:
            county += ", " + state
        return county

    def _get_state(self, display_name: str) -> str:
        parts = display_name.split(",")
        if len(parts) < 2:
            return ""

        # state is second last part of display name unless that is a zip code, in which case state is the one before
        state = parts[-2].strip()
        if state.isdigit():
            state = display_name.split(",")[-3].strip()
        return state
