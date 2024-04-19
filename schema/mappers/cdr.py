import logging

from schema.cdr_schemas.georeference import (
    GeoreferenceResults as CDRGeoreferenceResults,
    GroundControlPoint,
    Geom_Point,
    Pixel_Point,
    GeoreferenceResult,
    ProjectionResult,
)
from schema.cdr_schemas.area_extraction import Area_Extraction, AreaType
from schema.cdr_schemas.metadata import MapMetaData, CogMetaData
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.features.point_features import (
    PointFeatureCollection,
    PointLegendAndFeaturesResult,
    PointFeature,
    Point,
    PointProperties,
)

from tasks.geo_referencing.entities import GeoreferenceResult as LARAGeoferenceResult
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata
from tasks.point_extraction.entities import MapImage as LARAPoints
from tasks.segmentation.entities import MapSegmentation as LARASegmentation

from pydantic import BaseModel

from typing import Dict, List

logger = logging.getLogger("mapper")

MODEL_NAME = "uncharted-lara"
MODEL_VERSION = "0.0.1"


class CDRMapper:

    def __init__(self, system_name: str, system_version: str):
        self._system_name = system_name
        self._system_version = system_version

    def map_to_cdr(self, model: BaseModel) -> BaseModel:
        raise NotImplementedError()

    def map_from_cdr(self, model: BaseModel) -> BaseModel:
        raise NotImplementedError()


class GeoreferenceMapper(CDRMapper):

    def map_to_cdr(self, model: LARAGeoferenceResult) -> CDRGeoreferenceResults:
        gcps = []
        for gcp in model.gcps:
            cdr_gcp = GroundControlPoint(
                gcp_id=gcp.id,
                map_geom=Geom_Point(latitude=gcp.latitude, longitude=gcp.longitude),
                px_geom=Pixel_Point(
                    rows_from_top=gcp.pixel_y, columns_from_left=gcp.pixel_x
                ),
                confidence=gcp.confidence,
                model=MODEL_NAME,
                model_version=MODEL_VERSION,
                crs=model.projection,
            )
            gcps.append(cdr_gcp)

        return CDRGeoreferenceResults(
            cog_id=model.map_id,
            georeference_results=[
                GeoreferenceResult(
                    likely_CRSs=[model.projection],
                    map_area=None,
                    projections=[
                        ProjectionResult(
                            crs=model.projection,
                            gcp_ids=[gcp.gcp_id for gcp in gcps],
                            file_name=f"lara-{model.map_id}.tif",
                        )
                    ],
                )
            ],
            gcps=gcps,
            system=self._system_name,
            system_version=self._system_version,
        )

    def map_from_cdr(self, model: CDRGeoreferenceResults) -> LARAGeoferenceResult:
        raise NotImplementedError()


class MetadataMapper(CDRMapper):

    def map_to_cdr(self, model: LARAMetadata) -> CogMetaData:
        scale_str = "0"
        if model.scale:
            scale_split = model.scale.split(":")
            if len(scale_split) > 1:
                scale_str = scale_split[1]
        return CogMetaData(
            cog_id=model.map_id,
            system=self._system_name,
            system_version=self._system_version,
            multiple_maps=False,
            map_metadata=[
                MapMetaData(
                    title=model.title,
                    year=int(model.year),
                    scale=int(scale_str),
                    authors=model.authors,
                    quadrangle_name=",".join(model.quadrangles),
                    map_shape=None,
                    map_color_scheme=None,
                    state=",".join(model.states),
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                ),
            ],
        )

    def map_from_cdr(self, model: CogMetaData) -> LARAMetadata:
        raise NotImplementedError()


class SegmentationMapper(CDRMapper):
    AREA_MAPPING = {
        "cross_section": AreaType.CrossSection,
        "legend_points_lines": AreaType.Line_Point_Legend_Area,
        "legend_polygons": AreaType.Polygon_Legend_Area,
        "map": AreaType.Map_Area,
    }

    def map_to_cdr(self, model: LARASegmentation) -> FeatureResults:
        area_extractions: List[Area_Extraction] = []
        # create CDR area extractions for segment we've identified in the map
        for i, segment in enumerate(model.segments):
            coordinates = [list(point) for point in segment.poly_bounds]

            if segment.class_label in SegmentationMapper.AREA_MAPPING:
                area_type = SegmentationMapper.AREA_MAPPING[segment.class_label]
            else:
                logger.warning(
                    f"Unknown area type {segment.class_label} for map {model.doc_id}"
                )
                area_type = AreaType.Map_Area

            area_extraction = Area_Extraction(
                coordinates=[coordinates],
                bbox=segment.bbox,
                category=area_type,
                confidence=segment.confidence,  # assume two points - ll, ur
                model=MODEL_NAME,
                model_version=MODEL_VERSION,
            )
            area_extractions.append(area_extraction)

        return FeatureResults(
            # relevant to segment extractions
            cog_id=model.doc_id,
            cog_area_extractions=area_extractions,
            system=self._system_name,
            system_version=self._system_version,
        )

    def map_from_cdr(self, model: FeatureResults) -> LARASegmentation:
        raise NotImplementedError()


class PointsMapper(CDRMapper):

    def map_to_cdr(self, model: LARAPoints) -> FeatureResults:
        point_features: List[PointLegendAndFeaturesResult] = []

        # create seperate lists for each point class since they are groupded by class
        # in the results
        point_features_by_class: Dict[str, List[PointFeature]] = {}
        point_id = 0
        if model.labels:
            for map_pt_label in model.labels:
                if map_pt_label.class_name not in point_features_by_class:
                    point_features_by_class[map_pt_label.class_name] = []

                # create the point geometry
                point = Point(
                    coordinates=[
                        (map_pt_label.x1 + map_pt_label.x2) / 2,
                        (map_pt_label.y1 + map_pt_label.y2) / 2,
                    ]
                )

                # create the additional point properties
                properties = PointProperties(
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                    confidence=map_pt_label.score,
                    bbox=[
                        map_pt_label.x1,
                        map_pt_label.y1,
                        map_pt_label.x2,
                        map_pt_label.y2,
                    ],
                    dip=round(map_pt_label.dip) if map_pt_label.dip else None,
                    dip_direction=(
                        round(map_pt_label.direction)
                        if map_pt_label.direction
                        else None
                    ),
                )

                # add the point geometry and properties to the point feature
                point_feature = PointFeature(
                    id=f"{map_pt_label.class_id}.{point_id}",
                    geometry=point,
                    properties=properties,
                )
                point_id += 1

                # add to the list of point features for the class
                point_features_by_class[map_pt_label.class_name].append(point_feature)

        # create a PointLegendAndFeaturesResult for each class - we don't use the legend
        # so the legend bbox is empty
        for class_name, features in point_features_by_class.items():
            point_features_result = PointLegendAndFeaturesResult(
                id="id",
                crs="CRITICALMAAS:pixel",
                name=class_name,
                point_features=PointFeatureCollection(features=features),
            )

            # add to our final list of features results and create the output
            point_features.append(point_features_result)

        return FeatureResults(
            cog_id=model.raster_id,
            point_feature_results=point_features,
            system=MODEL_NAME,
            system_version=MODEL_VERSION,
        )

    def map_from_cdr(self, model: FeatureResults) -> LARAPoints:
        raise NotImplementedError()


def get_mapper(model: BaseModel, system_name: str, system_version: str) -> CDRMapper:
    if isinstance(model, LARAGeoferenceResult):
        return GeoreferenceMapper(system_name, system_version)
    elif isinstance(model, LARAMetadata):
        return MetadataMapper(system_name, system_version)
    elif isinstance(model, LARASegmentation):
        return SegmentationMapper(system_name, system_version)
    elif isinstance(model, LARAPoints):
        return PointsMapper(system_name, system_version)
    raise Exception("mapping does not support the requested type")
