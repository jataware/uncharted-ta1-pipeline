from schema.cdr_schemas.georeference import (
    GeoreferenceResults as CDRGeoreferenceResults,
    GroundControlPoint,
    Geom_Point,
    Pixel_Point,
    GeoreferenceResult,
    ProjectionResult,
)
from schema.cdr_schemas.metadata import MapMetaData, CogMetaData

from tasks.geo_referencing.entities import GeoreferenceResult as LARAGeoferenceResult
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata

from pydantic import BaseModel

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
        return CogMetaData(
            cog_id=model.map_id,
            system=self._system_name,
            system_version=self._system_version,
            multiple_maps=False,
            map_metadata=[
                MapMetaData(
                    title=model.title,
                    year=int(model.year),
                    scale=int(model.scale.split(":")[1]),
                    crs=None,
                    authors=model.authors,
                    organization=None,
                    quadrangle_name=",".join(model.quadrangles),
                    map_shape=None,
                    map_color_scheme=None,
                    publisher=None,
                    state=",".join(model.states),
                    model=MODEL_NAME,
                    model_version=MODEL_VERSION,
                ),
            ],
        )

    def map_from_cdr(self, model: CogMetaData) -> LARAMetadata:
        raise NotImplementedError()


def get_mapper(model: BaseModel, system_name: str, system_version: str) -> CDRMapper:
    if isinstance(model, LARAGeoferenceResult):
        return GeoreferenceMapper(system_name, system_version)
    elif isinstance(model, LARAMetadata):
        return MetadataMapper(system_name, system_version)
    raise Exception("mapping does not support the requested type")
