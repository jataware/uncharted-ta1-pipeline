from math import pi
import jsons
import os
import uuid

import pip

from tasks.common.pipeline import (
    BaseModelOutput,
    ObjectOutput,
    Output,
    TabularOutput,
    OutputCreator,
    PipelineResult,
)
from tasks.geo_referencing.entities import (
    GeoreferenceResult,
    GroundControlPoint as LARAGroundControlPoint,
    SOURCE_GEOCODE,
    SOURCE_LAT_LON,
    SOURCE_STATE_PLANE,
    SOURCE_UTM,
    SOURCE_INFERENCE,
)

from typing import Any, Dict, List

from tasks.geo_referencing.georeference import QueryPoint


class GeoReferencingOutput(OutputCreator):
    def __init__(self, id: str, extended: bool = False):
        super().__init__(id)
        self._extended = extended

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = TabularOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)
        res.data = []
        res.fields = [
            "raster_id",
            "row",
            "col",
            "NAD83_x",
            "NAD83_y",
        ]
        if self._extended:
            res.fields = res.fields + [
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
            }
            if self._extended and qp.lonlat_gtruth:
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
                o["confidence"] = qp.confidence
            res.data.append(o)
        return res


class SummaryOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = TabularOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)
        res.fields = [
            "raster_id",
            "lat - lat/lon",
            "lat - utm",
            "lat - state plane",
            "lat - geocode",
            "lat - infer",
            "lat - anchor",
            "lon - lat/lon",
            "lon - utm",
            "lon - state plane",
            "lon - geocode",
            "lon - infer",
            "lon - anchor",
            "rmse",
            "error_scale",
            "confidence",
        ]

        # reduce the keypoint counts to the correct numbers
        stats = [0] * 12
        if "keypoints" in pipeline_result.data:
            keypoints = pipeline_result.data["keypoints"]
            if "lats" in keypoints:
                lats = keypoints["lats"]
                stats[0] = lats[SOURCE_LAT_LON] if SOURCE_LAT_LON in lats else 0
                stats[1] = lats[SOURCE_UTM] if SOURCE_UTM in lats else 0
                stats[2] = lats[SOURCE_STATE_PLANE] if SOURCE_STATE_PLANE in lats else 0
                stats[3] = lats[SOURCE_GEOCODE] if SOURCE_GEOCODE in lats else 0
                stats[4] = lats[SOURCE_INFERENCE] if SOURCE_INFERENCE in lats else 0
                stats[5] = lats["anchor"] if "anchor" in lats else 0
            if "lons" in keypoints:
                lons = keypoints["lons"]
                stats[6] = lons[SOURCE_LAT_LON] if SOURCE_LAT_LON in lons else 0
                stats[7] = lons[SOURCE_UTM] if SOURCE_UTM in lons else 0
                stats[8] = lons[SOURCE_STATE_PLANE] if SOURCE_STATE_PLANE in lons else 0
                stats[9] = lons[SOURCE_GEOCODE] if SOURCE_GEOCODE in lons else 0
                stats[10] = lons[SOURCE_INFERENCE] if SOURCE_INFERENCE in lons else 0
                stats[11] = lons["anchor"] if "anchor" in lons else 0

        # obtain the rmse and other summary output
        res.data = [
            {
                "raster_id": pipeline_result.raster_id,
                "lat - lat/lon": stats[0],
                "lat - utm": stats[1],
                "lat - state plane": stats[2],
                "lat - geocode": stats[3],
                "lat - infer": stats[4],
                "lat - anchor": stats[5],
                "lon - lat/lon": stats[6],
                "lon - utm": stats[7],
                "lon - state plane": stats[8],
                "lon - geocode": stats[9],
                "lon - infer": stats[10],
                "lon - anchor": stats[11],
                "rmse": pipeline_result.data["rmse"],
                "error_scale": pipeline_result.data["error_scale"],
                "confidence": pipeline_result.data["query_pts"][0].confidence,
            }
        ]
        return res


class UserLeverOutput(OutputCreator):
    def __init__(self, id: str):
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
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None
        crs = pipeline_result.data["crs"]
        query_points: List[QueryPoint] = pipeline_result.data["query_pts"]

        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        res.data = {
            "map": pipeline_result.raster_id,
            "crs": [crs],
            "image_height": pipeline_result.image.size[1],
            "image_width": pipeline_result.image.size[0],
            "gcps": [],
            "levers": [],
        }

        for qp in query_points:
            o = {
                "crs": crs,
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


class LARAModelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        # capture query points as output
        query_points = pipeline_result.data["query_pts"]
        crs = pipeline_result.data["crs"]
        confidence = 0

        gcps: List[LARAGroundControlPoint] = []

        result = GeoreferenceResult(
            map_id=pipeline_result.raster_id,
            gcps=gcps,
            crs=crs,
            provenance="modelled",
            confidence=confidence,
        )

        for i, qp in enumerate(query_points):
            o = LARAGroundControlPoint(
                id=f"gcp.{i}",
                pixel_x=qp.xy[0],
                pixel_y=qp.xy[1],
                latitude=qp.lonlat[1],
                longitude=qp.lonlat[0],
                confidence=qp.confidence,
            )
            gcps.append(o)

        result.gcps = gcps
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, result
        )


class CDROutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None
        query_points = pipeline_result.data["query_pts"]
        crs = pipeline_result.data["crs"]

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
                "crs": crs,
            }
            gcps.append(o)

        res.data = {
            "cog_id": pipeline_result.raster_id,
            "georeference_results": [
                {
                    "likely_CRSs": [crs],
                    "map_area": None,
                    "projections": [
                        {
                            "crs": crs,
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
