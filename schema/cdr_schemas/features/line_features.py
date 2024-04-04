from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cdr_schemas.common import GeoJsonType, GeomType


class DashType(str, Enum):
    solid = "solid"
    dash = "dash"
    dotted = "dotted"


class Line(BaseModel):
    """
    coordinates in line are  (column from left, row from bottom).
    """

    coordinates: List[List[Union[float, int]]]
    type: GeomType = GeomType.LineString


class LineProperty(BaseModel):
    """
    Properties of the line.
    """

    model: Optional[str] = Field(description="model name used for extraction")
    model_version: Optional[str] = Field(
        description="model version used for extraction"
    )
    confidence: Optional[float] = Field(
        description="The prediction probability from the ML model"
    )
    dash_pattern: Optional[DashType] = Field(
        default=None, description="values = {solid, dash, dotted}"
    )
    symbol: Optional[str]

    model_config = ConfigDict(protected_namespaces=())


class LineFeature(BaseModel):
    """
    Line Feature.
    """

    type: GeoJsonType = GeoJsonType.Feature
    id: str = Field(
        description="""Each line geometry has a unique id.
                    The ids are used to link the line geometries is px-coord and geo-coord."""
    )
    geometry: Line
    properties: LineProperty


class LineFeatureCollection(BaseModel):
    """
    All line features for legend item.
    """

    type: GeoJsonType = GeoJsonType.FeatureCollection
    features: Optional[List[LineFeature]]


class LineLegendAndFeaturesResult(BaseModel):
    """
    Line legend item with metadata and associated line features found.
    """

    id: str = Field(description="your internal id")
    crs: str = Field(description="values={CRITICALMAAS:pixel, EPSG:*}")
    cdr_projection_id: Optional[str] = Field(
        description="""
                        A cdr projection id used to georeference the features
                    """
    )
    name: Optional[str]
    description: Optional[str]
    legend_bbox: Optional[List[Union[float, int]]] = Field(
        description="""The extacted bounding box of the legend item.
        Column value from left, row value from bottom."""
    )
    line_features: Optional[LineFeatureCollection]
