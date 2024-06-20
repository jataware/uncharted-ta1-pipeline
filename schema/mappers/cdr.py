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
from schema.cdr_schemas.map import Map
from schema.cdr_schemas.metadata import (
    MapColorSchemeTypes,
    MapMetaData,
    CogMetaData,
    MapShapeTypes,
)
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
    MapChromaType,
    MapShape,
    MetadataExtraction as LARAMetadata,
)
from tasks.point_extraction.entities import MapImage as LARAPoints
from tasks.point_extraction.label_map import YOLO_TO_CDR_LABEL
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

        # map the chrom to the CDR map color scheme - CDR has different
        # values (monochrome, full color, grayscale) that I think are probably
        # not what we want in there.  We'll just map to full color / monochrome
        # for now
        cdr_map_color_scheme = None
        match model.map_chroma:
            case MapChromaType.UNKNOWN:
                cdr_map_color_scheme = None
            case MapChromaType.LOW_CHROMA:
                cdr_map_color_scheme = MapColorSchemeTypes.full_color
            case MapChromaType.MONO_CHROMA:
                cdr_map_color_scheme = MapColorSchemeTypes.monochrome
            case MapChromaType.HIGH_CHROMA:
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

    def map_to_cdr(self, model: LARAPoints) -> FeatureResults:
        point_features: List[PointLegendAndFeaturesResult] = []

        # create seperate lists for each point class since they are groupded by class
        # in the results
        point_features_by_class: Dict[str, List[PointFeature]] = {}
        point_id = 0
        if model.labels:
            for map_pt_label in model.labels:
                # point label
                pt_label = (
                    map_pt_label.legend_name
                    if map_pt_label.legend_name
                    else map_pt_label.class_name
                )

                if pt_label in YOLO_TO_CDR_LABEL:
                    # map from YOLO point class to CDR point label
                    pt_label = YOLO_TO_CDR_LABEL[pt_label]

                if pt_label not in point_features_by_class:
                    # init result object for this point type...
                    # TODO -- fill in legend item info if available in future
                    point_features_by_class[pt_label] = []
                    point_features_result = PointLegendAndFeaturesResult(
                        id="id",
                        crs="CRITICALMAAS:pixel",
                        name=pt_label,
                        abbreviation=pt_label,
                        description=pt_label.replace("_", " ").strip().lower(),
                        legend_bbox=map_pt_label.legend_bbox,
                        point_features=None,  # points are filled in below
                    )
                    point_features.append(point_features_result)

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
                    id=f"{pt_label}.{point_id}",
                    geometry=point,
                    properties=properties,
                )
                point_id += 1

                # add to the list of point features for the class
                point_features_by_class[pt_label].append(point_feature)

        # append our final list of feature results and create the output
        for pt_feat in point_features:
            if pt_feat.name not in point_features_by_class:
                logger.warning(f"Point type {pt_feat.name} not found in results!")
            else:
                pt_feat.point_features = PointFeatureCollection(
                    features=point_features_by_class[pt_feat.name]
                )

        return FeatureResults(
            cog_id=model.raster_id,
            point_feature_results=point_features,
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
