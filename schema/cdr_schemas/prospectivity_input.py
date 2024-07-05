from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class InterpolationType(str, Enum):
    """Enum for the possible values of type field of MapUnit"""

    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"
    NONE = "none"


class ScalingType(str, Enum):
    """Enum for the possible values of type field of MapUnit"""

    MINMAX = "minmax"
    MAXABS = "maxabs"
    STANDARD = "standard"


class LayerCategory(str, Enum):
    GEOPHYSICS = "geophysics"
    GEOLOGY = "geology"
    GEOCHEMISTRY = "geochemistry"


class LayerDataType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class DataFormat(str, Enum):
    TIF = "tif"
    SHP = "shp"


class DataSource(BaseModel):
    DOI: Optional[str]
    authors: Optional[List[str]]
    publication_date: Optional[str]
    category: Optional[Union[LayerCategory, str]]
    subcategory: Optional[str]
    description: Optional[str]
    derivative_ops: Optional[str]
    type: LayerDataType
    resolution: Optional[tuple]
    format: DataFormat
    download_url: Optional[str]


class ProcessedDataLayer(BaseModel):
    title: Optional[str]
    resampling_method: InterpolationType
    scaling_method: ScalingType
    normalization_method: str  # source: LayerDataType


class StackMetaData(BaseModel):
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
            Creators of the dataset
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
            Mean scale of the map. 24000 would be equivalent to 1:24000.
        """,
    )

    evidence_layers: List[ProcessedDataLayer]
