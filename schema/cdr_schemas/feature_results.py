from typing import List, Optional

from pydantic import BaseModel, Field

from schema.cdr_schemas.area_extraction import Area_Extraction
from schema.cdr_schemas.features.line_features import LineLegendAndFeaturesResult
from schema.cdr_schemas.features.point_features import PointLegendAndFeaturesResult
from schema.cdr_schemas.features.polygon_features import PolygonLegendAndFeauturesResult
from schema.cdr_schemas.metadata import CogMetaData


class FeatureResults(BaseModel):
    """
    Feature Extraction Results.
    """

    cog_id: str = Field(
        ...,
        description="""
            Cog id.
        """,
    )
    line_feature_results: Optional[List[LineLegendAndFeaturesResult]] = Field(
        ...,
        description="""
            A list of legend extractions with associated line feature results.
        """,
    )
    point_feature_results: Optional[List[PointLegendAndFeaturesResult]] = Field(
        ...,
        description="""
            A list of legend extractions with associated point feature results.
        """,
    )
    polygon_feature_results: Optional[List[PolygonLegendAndFeauturesResult]] = Field(
        ...,
        description="""
            A list of legend extractions with associated polygon feature results.
        """,
    )
    cog_area_extractions: Optional[List[Area_Extraction]]
    cog_metadata_extractions: Optional[List[CogMetaData]]
    system: str = Field(
        ...,
        description="""
            The name of the system used.
        """,
    )
    system_version: str = Field(
        ...,
        description="""
            The version of the system used.
        """,
    )
