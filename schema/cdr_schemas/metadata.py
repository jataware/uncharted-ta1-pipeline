from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MapShapeTypes(str, Enum):
    """Enum for the possible values of map_shape field of MapMetadata."""

    rectangular = "rectangular"
    non_rectangular = "non_rectangular"


class MapColorSchemeTypes(str, Enum):
    """Enum for the possible values of map_color_scheme field of MapMetadata"""

    full_color = "full_color"
    monochrome = "monochrome"
    grayscale = "grayscale"


class MapMetaData(BaseModel):
    title: Optional[str] = Field(
        ...,
        description="""
            Title of the map/cog.
        """,
    )
    year: Optional[int] = Field(
        ...,
        description="""
            Year the map was made. i.e. 2012
        """,
    )
    crs: Optional[str] = Field(
        ...,
        description="""
            CRS of the map. i.e. "EPSG:4267"
        """,
    )
    authors: Optional[List[str]] = Field(
        ...,
        description="""
            Authors of the map
        """,
    )
    organization: Optional[str] = Field(
        ...,
        description="""
            Organization that created the map
        """,
    )
    scale: Optional[int] = Field(
        ...,
        description="""
            Scale of the map. 24000 would be equivalent to 1:24000
        """,
    )
    quadrangle_name: Optional[str] = Field(
        ...,
        description="""
            If map is based on a quadrangle location we can save the name here.
        """,
    )
    map_shape: Optional[MapShapeTypes] = Field(
        ...,
        description="""
            If the map area(s) has a rectangle shape.
        """,
    )
    map_color_scheme: Optional[MapColorSchemeTypes]
    publisher: Optional[str]
    state: Optional[str]

    model: str
    model_version: str

    model_config = ConfigDict(protected_namespaces=())


class CogMetaData(BaseModel):
    cog_id: str
    system: str
    system_version: str
    multiple_maps: Optional[bool]
    map_metadata: Optional[List[MapMetaData]]
