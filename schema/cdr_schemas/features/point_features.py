from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from cdr_schemas.common import GeoJsonType, GeomType


class Point(BaseModel):
    """
    coordinates in line are (column from left, row from bottom).
    """

    coordinates: List[Union[float, int]]
    type: GeomType = GeomType.Point


class PointProperties(BaseModel):
    """
    Properties of the Point.
    """

    model: Optional[str] = Field(description="model name used for extraction")
    model_version: Optional[str] = Field(
        description="model version used for extraction"
    )
    confidence: Optional[float] = Field(
        description="The prediction probability from the ML model"
    )
    bbox: Optional[List[Union[float, int]]] = Field(
        description="""The extacted bounding box of the point item.
        Column value from left, row value from bottom."""
    )
    dip: Optional[int]
    dip_direction: Optional[int]

    model_config = ConfigDict(protected_namespaces=())


class PointFeature(BaseModel):
    """
    Point feature.
    """

    type: GeoJsonType = GeoJsonType.Feature
    id: str = Field(
        description="""Each point geometry has a unique id.
                    The ids are used to link the point geometries is px-coord and geo-coord."""
    )
    geometry: Point
    properties: PointProperties


class PointFeatureCollection(BaseModel):
    """
    All point features for legend item.
    """

    type: GeoJsonType = GeoJsonType.FeatureCollection
    features: List[PointFeature]


class PointLegendAndFeaturesResult(BaseModel):
    """
    Point legend item metadata along with associated point features found.
    """

    id: str = Field(description="your internal id")
    crs: str = Field(description="values={CRITICALMAAS:pixel, EPSG:*}")
    cdr_projection_id: Optional[str] = Field(
        description="""
                        A cdr projection id used to georeference the features
                    """
    )
    name: Optional[str] = Field(description="name of legend item")
    description: Optional[str]
    legend_bbox: Optional[List[Union[float, int]]] = Field(
        description="""
        The extacted bounding box of the legend item.
        Column value from left, row value from bottom.
        """
    )
    point_features: Optional[List[PointFeatureCollection]]
