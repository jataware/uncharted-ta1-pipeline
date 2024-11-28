from enum import Enum
from tasks.text_extraction.entities import TextExtraction

from typing import List, Tuple
from pydantic import BaseModel, ConfigDict


METADATA_EXTRACTION_OUTPUT_KEY = "metadata_extraction_output"
GEOCODED_PLACES_OUTPUT_KEY = "geocoded_places_output"


class MapShape(str, Enum):
    RECTANGULAR = "rectangular"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class MapColorType(str, Enum):
    MONO = "mono"
    LOW = "low"
    HIGH = "high"
    UNKNOWN = "unknown"


class GeoPlaceType(str, Enum):
    BOUND = "bound"
    POPULATION = "population"
    POINT = "point"


class GeoFeatureType(str, Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"
    CITY = "city"
    SETTLEMENT = "settlement"
    NOT_SET = ""


class MetadataExtraction(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    map_id: str
    title: str
    authors: List[str]
    year: str  # should be an int, but there's a chance somethign else is (incorrectly) extracted
    scale: str  # of the format 1:24000
    quadrangles: List[str]
    datum: str
    projection: str
    coordinate_systems: List[str]
    crs: str
    utm_zone: str
    base_map: str
    counties: List[str]
    population_centres: (
        List[TextExtraction] | List[str]
    )  # a list of cities, towns, and villages
    states: List[str]
    country: str
    places: (
        List[TextExtraction] | List[str]
    )  # a list of places, each place having a name and coordinates
    publisher: str
    map_shape: MapShape
    map_color: MapColorType
    language: str
    language_country: str


class GeocodedCoordinate(BaseModel):
    pixel_x: float
    pixel_y: float
    geo_x: float
    geo_y: float


class GeocodedResult(BaseModel):
    place_region: str  # second administrative division (ex: state)
    coordinates: List[
        GeocodedCoordinate
    ]  # bounds will not have the pixel coordinates defined, with each geocoding option a separate list of coordinates


class GeocodedPlace(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    place_name: str
    parent_country: str  # restrict the search space when geocoding to the parent country to reduce noise
    place_type: GeoPlaceType  # either bound for places that are not on the map but restrict the coordinate space, or point / line / polygon
    feature_type: GeoFeatureType  # type of place (country, state, county, etc)
    results: List[
        GeocodedResult
    ]  # bounds will not have the pixel coordinates defined, with each geocoding option a separate list of coordinates


class DocGeocodedPlaces(BaseModel):
    map_id: str
    places: List[GeocodedPlace] = []
