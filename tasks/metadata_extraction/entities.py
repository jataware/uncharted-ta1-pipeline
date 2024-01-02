from typing import List
from pydantic import BaseModel, ConfigDict


METADATA_EXTRACTION_OUTPUT_KEY = "metadata_extraction_output"


class MetadataExtraction(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    map_id: str
    title: str
    authors: List[str]
    year: str  # should be an int, but there's a chance somethign else is (incorrectly) extracted
    scale: str  # of the format 1:24000
    quadrangles: List[str]
    datum: str
    vertical_datum: str
    projection: str
    coordinate_systems: List[str]
    base_map: str
    counties: List[str]
    states: List[str]
    country: str
