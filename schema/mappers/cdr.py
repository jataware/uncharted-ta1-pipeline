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
from schema.cdr_schemas.metadata import (
    MapColorSchemeTypes,
    MapMetaData,
    CogMetaData,
    MapShapeTypes,
)

from schema.cdr_schemas.common import ModelProvenance

from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.features.point_features import (
    PointFeatureCollection,
    PointLegendAndFeaturesResult,
    PointFeature,
    Point,
    PointProperties,
)

from tasks.geo_referencing.entities import GeoreferenceResult as LARAGeoferenceResult
from tasks.metadata_extraction.entities import (
    MapColorType,
    MapShape,
    MetadataExtraction as LARAMetadata,
)
from tasks.point_extraction.entities import PointLabels as LARAPoints
from tasks.point_extraction.entities import LegendPointItems as LARALegendItems
from tasks.point_extraction.point_extractor_utils import get_cdr_point_name
from tasks.segmentation.entities import MapSegmentation as LARASegmentation

from pydantic import BaseModel

from typing import Dict, List

logger = logging.getLogger("mapper")

MODEL_NAME = "uncharted-lara"
MODEL_VERSION = "0.0.4"


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
    """
    Mapper class for converting between LARAMetadata and CogMetaData objects.
    """

    def map_to_cdr(self, model: LARAMetadata) -> CogMetaData:
        """
        Maps the given LARAMetadata object to a CogMetaData object.

        Args:
            model (LARAMetadata): The LARAMetadata object to be mapped.

        Returns:
            CogMetaData: The mapped CogMetaData object.
        """

        # extract the scale from the model
        scale_str = "0"
        if model.scale:
            scale_split = model.scale.split(":")
            if len(scale_split) > 1:
                scale_str = scale_split[1]

        # attempt to extract the year from the model
        try:
            year_int = int(model.year)
        except ValueError:
            year_int = None

        # map the map shape to the CDR map shape
        cdr_map_shape = None
        match model.map_shape:
            case MapShape.UNKNOWN:
                cdr_map_shape = None
            case MapShape.IRREGULAR:
                cdr_map_shape = MapShapeTypes.non_rectangular
            case MapShape.RECTANGULAR:
                cdr_map_shape = MapShapeTypes.rectangular

        # map the map color level to the CDR map color scheme - CDR has different
        # values (monochrome, full color, grayscale) that I think are probably
        # not what we want in there.  We'll just map to full color / monochrome
        # for now
        cdr_map_color_scheme = None
        match model.map_color:
            case MapColorType.UNKNOWN:
                cdr_map_color_scheme = None
            case MapColorType.LOW:
                cdr_map_color_scheme = MapColorSchemeTypes.full_color
            case MapColorType.MONO:
                cdr_map_color_scheme = MapColorSchemeTypes.monochrome
            case MapColorType.HIGH:
                cdr_map_color_scheme = MapColorSchemeTypes.full_color

        return CogMetaData(
            cog_id=model.map_id,
            system=self._system_name,
            system_version=self._system_version,
            multiple_maps=False,
            map_metadata=[
                MapMetaData(
                    title=model.title,
                    year=year_int,
                    scale=int(scale_str),
                    authors=model.authors,
                    quadrangle_name=",".join(model.quadrangles),
                    map_shape=cdr_map_shape,
                    map_color_scheme=cdr_map_color_scheme,
                    state=",".join(model.states),
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                    publisher=model.publisher,
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

            bbox = [
                segment.bbox[0],
                segment.bbox[1],
                segment.bbox[0] + segment.bbox[2],
                segment.bbox[1] + segment.bbox[3],
            ]

            area_extraction = Area_Extraction(
                coordinates=[coordinates],
                bbox=bbox,
                category=area_type,
                confidence=segment.confidence,
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

    def map_to_cdr(
        self, model: LARAPoints, legend_items: LARALegendItems
    ) -> FeatureResults:
        """
        Convert LARA point extractions to CDR output format
        """
        legend_features: Dict[str, PointLegendAndFeaturesResult] = {}
        for leg_item in legend_items.items:
            # fill in point name, abbreviation and description fields
            name = get_cdr_point_name(leg_item.class_name, leg_item.name)
            abbr = leg_item.abbreviation if leg_item.abbreviation else name
            desc = leg_item.description
            if not desc:
                desc = leg_item.name if leg_item.name else name
                desc = desc.replace("_", " ").strip().lower()

            legend_provenance = None
            if leg_item.system:
                legend_provenance = ModelProvenance(
                    model=leg_item.system, model_version=leg_item.system_version
                )

            legend_feature_result = PointLegendAndFeaturesResult(
                id=name,
                legend_provenance=legend_provenance,
                crs="CRITICALMAAS:pixel",
                name=name,
                abbreviation=abbr,
                description=desc,
                legend_bbox=leg_item.legend_bbox,
                legend_contour=leg_item.legend_contour,
                reference_id=leg_item.cdr_legend_id,
                point_features=None,  # points are filled in below
            )
            legend_features[name] = legend_feature_result

        # create seperate lists for each point class since they are grouped by class
        # in the results
        point_features: Dict[str, List[PointFeature]] = {}
        point_id = 0
        if model.labels:
            for map_pt_label in model.labels:

                name = get_cdr_point_name(map_pt_label.class_name, "")

                if name not in legend_features:
                    # init new legend result object...
                    legend_feature_result = PointLegendAndFeaturesResult(
                        id=name,
                        crs="CRITICALMAAS:pixel",
                        name=name,
                        abbreviation=name,
                        description=name.replace("_", " ").strip().lower(),
                        point_features=None,  # points are filled in below
                    )
                    legend_features[name] = legend_feature_result

                if name not in point_features:
                    # init result object for this point type...
                    point_features[name] = []

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
                    id=f"{name}.{point_id}",
                    geometry=point,
                    properties=properties,
                )
                point_id += 1

                # add to the list of point features for the class
                point_features[name].append(point_feature)

        # append our final list of feature results and create the output
        point_legend_features = list(legend_features.values())
        for pt_leg_feat in point_legend_features:
            if (
                pt_leg_feat.name not in point_features
                or len(point_features[pt_leg_feat.name]) == 0
            ):
                logger.warning(f"Point type {pt_leg_feat.name} has no extractions!")
            else:
                pt_leg_feat.point_features = PointFeatureCollection(
                    features=point_features[pt_leg_feat.name]
                )

        # filter point types with no extractions
        point_legend_features = list(
            filter(lambda x: x.point_features is not None, point_legend_features)
        )

        return FeatureResults(
            cog_id=model.raster_id,
            point_feature_results=point_legend_features,
            system=self._system_name,
            system_version=self._system_version,
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
