import jsons
import os
from pathlib import Path
import uuid

from tasks.common.pipeline import (
    BaseModelOutput,
    GeopackageOutput,
    ObjectOutput,
    Output,
    TabularOutput,
    OutputCreator,
    PipelineResult,
)
from schema.ta1_schema import (
    Map,
    GeoReferenceMeta,
    GroundControlPoint,
    MapFeatureExtractions,
    MapMetadata,
    ProvenanceType,
)
from shapely import Polygon, Point
from pandas import DataFrame
from geopandas import GeoDataFrame
from criticalmaas.ta1_geopackage import GeopackageDatabase

from typing import Any, Dict, List


def get_projection(datum: str) -> str:
    # get espg code via basic lookup of the two frequently seen datums
    if "83" in datum:
        return "EPSG:4269"
    elif "27" in datum:
        return "EPSG:4267"
    return "EPSG:4326"


class GeoReferencingOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = TabularOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)
        res.data = []
        res.fields = [
            "raster_id",
            "row",
            "col",
            "NAD83_x",
            "NAD83_y",
            "actual_x",
            "actual_y",
            "error_x",
            "error_y",
            "Distance",
            "error_scale",
            "pixel_dist_xx",
            "pixel_dist_xy",
            "pixel_dist_x",
            "pixel_dist_yx",
            "pirxel_dist_yy",
            "pixel_dist_y",
            "confidence",
        ]

        if "query_pts" not in pipeline_result.data:
            return res

        query_points = pipeline_result.data["query_pts"]
        for qp in query_points:
            o = {
                "raster_id": pipeline_result.raster_id,
                "row": qp.xy[1],
                "col": qp.xy[0],
                "NAD83_x": qp.lonlat[0],
                "NAD83_y": qp.lonlat[1],
                "confidence": qp.confidence,
            }
            if qp.lonlat_gtruth:
                o["actual_x"] = qp.lonlat_gtruth[0]
                o["actual_y"] = qp.lonlat_gtruth[1]
                o["error_x"] = qp.error_lonlat[0]
                o["error_y"] = qp.error_lonlat[1]
                o["distance"] = qp.error_km
                o["error_scale"] = qp.error_scale
                o["pixel_dist_xx"] = qp.lonlat_xp[0]
                o["pixel_dist_xy"] = qp.lonlat_xp[1]
                o["pixel_dist_x"] = qp.dist_xp_km
                o["pixel_dist_yx"] = qp.lonlat_yp[0]
                o["pixel_dist_yy"] = qp.lonlat_yp[1]
                o["pixel_dist_y"] = qp.dist_yp_km
            res.data.append(o)
        return res


class SummaryOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = TabularOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)
        res.fields = [
            "raster_id",
            "keypoints",
            "lat_initial",
            "lon_initial",
            "lat",
            "lon",
            "extraction",
            "rmse",
            "error_scale",
            "confidence",
        ]

        # obtain the rmse and other summary output
        res.data = [
            {
                "raster_id": pipeline_result.raster_id,
                "keypoints": "",
                "lat_initial": "",
                "lon_initial": "",
                "lat": "",
                "lon": "",
                "extraction": "",
                "rmse": pipeline_result.data["rmse"],
                "error_scale": pipeline_result.data["error_scale"],
                "confidence": pipeline_result.data["query_pts"][0].confidence,
            }
        ]
        return res


class UserLeverOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        # extract the levers available via params
        res.data = {"raster_id": pipeline_result.raster_id, "levers": []}
        for p in pipeline_result.params:
            res.data["levers"].append(p)
        return res


class CSVWriter:
    def __init__(self):
        pass

    def output(self, output: List[TabularOutput], params: Dict[Any, Any] = {}):
        if len(output) == 0:
            return

        try:
            with open(params["path"], "w") as f_out:
                # write header line
                line_str = ",".join(output[0].fields)
                f_out.write(line_str + "\n")
                for r in output:
                    for d in r.data:
                        row = []
                        for f in r.fields:
                            if f in d:
                                row.append(f"{d[f]}")
                            else:
                                row.append("")
                        f_out.write(f'{",".join(row)}\n')

        except Exception as e:
            print("EXCEPTION saving results to CSV")
            print(repr(e))


class JSONWriter:
    def __init__(self):
        pass

    def output(self, output: List[ObjectOutput], params: Dict[Any, Any] = {}) -> str:
        output_target = []
        for o in output:
            output_target.append(o.data)
        json_raw = jsons.dumps(output_target, indent=4)

        # Writing to output file if path specified
        if "path" in params:
            file_path = params["path"]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f_out:
                f_out.write(json_raw)

        # return the raw json
        return json_raw


class GCPOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None
        query_points = pipeline_result.data["query_pts"]
        projection_raw = pipeline_result.data["projection"]
        datum_raw = pipeline_result.data["datum"]
        projection_mapped = get_projection(datum_raw)

        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        res.data = {
            "map": pipeline_result.raster_id,
            "crs": [projection_mapped],
            "datum_raw": datum_raw,
            "projection_raw": projection_raw,
            "image_height": pipeline_result.image.size[1],
            "image_width": pipeline_result.image.size[0],
            "gcps": [],
            "levers": [],
        }
        for qp in query_points:
            o = {
                "crs": projection_mapped,
                "gcp_id": uuid.uuid4(),
                "rowb": qp.xy[1],
                "coll": qp.xy[0],
                "x": qp.lonlat[0],
                "y": qp.lonlat[1],
            }
            if qp.properties and len(qp.properties) > 0:
                o["properties"] = qp.properties
            res.data["gcps"].append(o)

        # extract the levers available via params
        for p in pipeline_result.params:
            res.data["levers"].append(p)
        return res


class IntegrationOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        # capture query points as output
        query_points = pipeline_result.data["query_pts"]
        projection_raw = pipeline_result.data["projection"]
        datum_raw = pipeline_result.data["datum"]
        projection_mapped = get_projection(datum_raw)

        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        # need to format the data to meet the schema definition
        gcps = []
        count = 0
        for qp in query_points:
            count = count + 1
            o = {
                "id": count,
                "map_geom": (qp.lonlat[0], qp.lonlat[1]),
                "px_geom": (qp.xy[0], qp.xy[1]),
                "confidence": qp.confidence,
                "provenance": "modelled",
            }
            gcps.append(o)
        res.data = {
            "map": {
                "name": pipeline_result.raster_id,
                "projection_info": {
                    "projection": projection_mapped,
                    "provenance": "modelled",
                    "gcps": gcps,
                },
            }
        }

        return res


class IntegrationModelOutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        # capture query points as output
        query_points = pipeline_result.data["query_pts"]
        projection_raw = pipeline_result.data["projection"]
        datum_raw = pipeline_result.data["datum"]
        projection_mapped = get_projection(datum_raw)

        gcps = []
        count = 0
        for qp in query_points:
            count = count + 1
            gcp = GroundControlPoint(
                id=f"gcp-{count}",
                map_geom=(qp.lonlat[0], qp.lonlat[1]),
                px_geom=(qp.xy[0], qp.xy[1]),
                confidence=qp.confidence,
                provenance=ProvenanceType.modelled,
            )
            gcps.append(gcp)

        schema_georeference = GeoReferenceMeta(
            gcps=gcps,
            projection=projection_mapped,
            bounds=None,
            provenance=ProvenanceType.modelled,
        )
        schema_metadata = MapMetadata(
            id=pipeline_result.raster_id,
            authors="",
            publisher="",
            year=-1,
            organization="",
            scale="",
            confidence=None,
            provenance=ProvenanceType.skipped,
        )
        schema_map = Map(
            name="",
            id=pipeline_result.raster_id,
            source_url="",
            image_url="",
            image_size=[],
            map_metadata=schema_metadata,
            features=MapFeatureExtractions(
                lines=[], points=[], polygons=[], pipelines=[]
            ),
            cross_sections=None,
            projection_info=schema_georeference,
        )
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, schema_map
        )


class GeopackageIntegrationOutput(OutputCreator):
    def __init__(self, id: str, output_dir: str):
        super().__init__(id)
        self._output_dir = output_dir

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a geopackage output from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            GeopackageOutput: A geopackage containing the metadata extraction results.
        """
        query_points = pipeline_result.data["query_pts"]
        projection_raw = pipeline_result.data["projection"]
        datum_raw = pipeline_result.data["datum"]
        projection_mapped = get_projection(datum_raw)
        os.makedirs(self._output_dir, exist_ok=True)

        filepath = os.path.join(self._output_dir, f"{pipeline_result.raster_id}.pkg")
        if os.path.exists(filepath):
            os.remove(filepath)

        db = GeopackageDatabase(filepath, crs="EPSG:4326")

        if db.model is None:
            raise ValueError("db.model is None")

        db.write_models(
            [
                db.model.map(
                    id=pipeline_result.raster_id,
                    name=pipeline_result.raster_id,
                    source_url="",
                    image_url="",
                    image_width=(
                        pipeline_result.image.width if pipeline_result.image else -1
                    ),
                    image_height=(
                        pipeline_result.image.height if pipeline_result.image else -1
                    ),
                )
            ]
        )

        geometa = dict(
            id=f"geometa-uncharted-{pipeline_result.raster_id}",
            map_id=pipeline_result.raster_id,
            projection=projection_mapped if projection_mapped else "",
            provenance="modelled",
            bounds=Polygon([]),
        )
        print([geometa])
        db.write_dataframe(
            GeoDataFrame([geometa], geometry="bounds"), "georeference_meta"
        )

        gcps = []
        count = 0
        for qp in query_points:
            count = count + 1
            # gcp = db.model.ground_control_point(
            #    id=f"gcp-uncharted-{count}",
            #    map_id=pipeline_result.raster_id,
            #    geometry=Point(qp.lonlat[0], qp.lonlat[1]),
            #    x=qp.xy[0],
            #    y=qp.xy[1],
            #    confidence=qp.confidence,
            #    provenance=ProvenanceType.modelled,
            # )
            gcp = dict(
                id=f"gcp-uncharted-{pipeline_result.raster_id}-{count}",
                map_id=pipeline_result.raster_id,
                geometry=Point(qp.lonlat[0], qp.lonlat[1]),
                x=qp.xy[0],
                y=qp.xy[1],
                confidence=qp.confidence,
                provenance="modelled",
            )
            # gcp = {
            #    "geometry": {"type": "Point", "coordinates": (int(qp.lonlat[0]), int(qp.lonlat[1]))},
            #    "properties": {
            #        "id": f"gcp-uncharted-{count}",
            #        "map_id": pipeline_result.raster_id,
            #        "x": qp.xy[0],
            #        "y": qp.xy[1],
            #        "confidence": qp.confidence,
            #        "provenance": ProvenanceType.modelled,
            #    },
            # "geometry": {"type": "Point", "coordinates": (qp.lonlat[0], qp.lonlat[1])},
            # }
            gcps.append(gcp)
        if gcps:
            print(gcps)
            db.write_dataframe(GeoDataFrame(gcps, geometry="geometry"), "ground_control_point")  # type: ignore
        # db.write_features("ground_control_point", gcps)
        # db.write_models(gcps)

        geometa = {
            "properties": {
                "id": f"geometa-uncharted-{pipeline_result.raster_id}",
                "map_id": pipeline_result.raster_id,
                "projection": projection_mapped if projection_mapped else "",
                "provenance": ProvenanceType.modelled,
                "bounds": {
                    "type": "Polygon",
                    "coordinates": [[0, 0], [0, 1], [1, 1], [1, 0]],
                },
            },
        }
        print([geometa])
        db.write_features("georeference_meta", [geometa])
        # models.append(db.model.georeference_meta(
        #    id=f"geometa-uncharted-{pipeline_result.raster_id}",
        #    map_id=pipeline_result.raster_id,
        #    projection=projection_mapped,
        #    bounds = Polygon([]),
        #    #bounds = {"type": "Polygon", "coordinates": [[[0, 0], [1,1], [0,0]]]},
        #    provenance=ProvenanceType.modelled
        # ))
        return GeopackageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
        )


class CDROutput(OutputCreator):
    def __init__(self, id):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None
        query_points = pipeline_result.data["query_pts"]
        projection_raw = pipeline_result.data["projection"]
        datum_raw = pipeline_result.data["datum"]
        projection_mapped = get_projection(datum_raw)

        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        gcps = []
        for qp in query_points:
            o = {
                "gcp_id": f"{len(gcps) + 1}",
                "map_geom": {"latitude": qp.lonlat[1], "longitude": qp.lonlat[0]},
                "px_geom": {"columns_from_left": qp.xy[0], "rows_from_top": qp.xy[1]},
                "confidence": qp.confidence,
                "model": "uncharted",
                "model_version": "0.0.1",
                "crs": projection_mapped,
            }
            gcps.append(o)

        res.data = {
            "cog_id": pipeline_result.raster_id,
            "georeference_results": [
                {
                    "likely_CRSs": [projection_mapped],
                    "map_area": None,
                    "projections": [
                        {
                            "crs": projection_mapped,
                            "gcp_ids": [gcp["gcp_id"] for gcp in gcps],
                            "file_name": f"lara-{pipeline_result.raster_id}.tif",
                        }
                    ],
                }
            ],
            "gcps": gcps,
            "system": "uncharted",
            "system_version": "0.0.1",
        }

        return res
