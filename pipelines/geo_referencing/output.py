import logging
from h11 import ERROR
import jsons
import os
import uuid
from typing import Any, Dict, List

from schema.mappers.cdr import GeoreferenceMapper
from tasks.common.pipeline import (
    BaseModelOutput,
    BytesOutput,
    ObjectOutput,
    Output,
    TabularOutput,
    OutputCreator,
    PipelineResult,
)
from tasks.geo_referencing.georeference import QueryPoint
from tasks.geo_referencing.entities import (
    CRS_OUTPUT_KEY,
    ERROR_SCALE_OUTPUT_KEY,
    KEYPOINTS_OUTPUT_KEY,
    LEVERS_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    RMSE_OUTPUT_KEY,
    GeoreferenceResult,
    GroundControlPoint as LARAGroundControlPoint,
    SOURCE_GEOCODE,
    SOURCE_LAT_LON,
    SOURCE_STATE_PLANE,
    SOURCE_UTM,
    SOURCE_INFERENCE,
)
from tasks.geo_referencing.util import cps_to_transform, project_image

logger = logging.getLogger(__name__)


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
            logger.error(f"Error writing CSV file: {e}", exc_info=True)


class JSONWriter:
    def __init__(self):
        pass

    def output(self, output: List[ObjectOutput], params: Dict[Any, Any] = {}) -> str:
        try:
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
        except Exception as e:
            logger.error(f"Error writing JSON file {e}", exc_info=True)
            return ""


class ScoringOutput(OutputCreator):
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

        if QUERY_POINTS_OUTPUT_KEY not in pipeline_result.data:
            return res

        query_points = pipeline_result.data[QUERY_POINTS_OUTPUT_KEY]
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
            "latlon",
            "utm",
            "state_plane",
            "geocode",
            "infer",
            "anchor",
            "rmse",
            "confidence",
        ]

        latlon = ""
        utm = ""
        state_plane = ""
        geocode = ""
        infer = ""
        anchor = ""

        if KEYPOINTS_OUTPUT_KEY in pipeline_result.data:
            keypoints = pipeline_result.data[KEYPOINTS_OUTPUT_KEY]

            lats = keypoints.get("lats", {})
            lons = keypoints.get("lons", {})

            if SOURCE_LAT_LON in lats or SOURCE_LAT_LON in lons:
                latlon = (
                    f"{lats.get(SOURCE_LAT_LON, '')};{lons.get(SOURCE_LAT_LON, '')}"
                )
            if SOURCE_UTM in lats or SOURCE_UTM in lons:
                utm = f"{lats.get(SOURCE_UTM, '')};{lons.get(SOURCE_UTM, '')}"
            if SOURCE_STATE_PLANE in lats or SOURCE_STATE_PLANE in lons:
                state_plane = f"{lats.get(SOURCE_STATE_PLANE, '')};{lons.get(SOURCE_STATE_PLANE, '')}"
            if SOURCE_GEOCODE in lats or SOURCE_GEOCODE in lons:
                geocode = (
                    f"{lats.get(SOURCE_GEOCODE, '')};{lons.get(SOURCE_GEOCODE, '')}"
                )
            if SOURCE_INFERENCE in lats or SOURCE_INFERENCE in lons:
                infer = (
                    f"{lats.get(SOURCE_INFERENCE, '')};{lons.get(SOURCE_INFERENCE, '')}"
                )
            if "anchor" in lats or "anchor" in lons:
                anchor = f"{lats.get('anchor', '')};{lons.get('anchor', '')}"

        res.data = [
            {
                "raster_id": pipeline_result.raster_id,
                "latlon": latlon,
                "utm": utm,
                "state_plane": state_plane,
                "geocode": geocode,
                "infer": infer,
                "anchor": anchor,
                "rmse": pipeline_result.data[RMSE_OUTPUT_KEY],
                "confidence": pipeline_result.data[QUERY_POINTS_OUTPUT_KEY][
                    0
                ].confidence,
            }
        ]
        return res


class UserLeverOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        # extract the levers available via params
        res.data = {"raster_id": pipeline_result.raster_id, LEVERS_OUTPUT_KEY: []}
        for p in pipeline_result.params:
            res.data[LEVERS_OUTPUT_KEY].append(p)
        return res


class GCPOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None
        crs = pipeline_result.data[CRS_OUTPUT_KEY]
        query_points: List[QueryPoint] = pipeline_result.data[QUERY_POINTS_OUTPUT_KEY]

        res = ObjectOutput(pipeline_result.pipeline_id, pipeline_result.pipeline_name)

        res.data = {
            "map": pipeline_result.raster_id,
            "crs": [crs],
            "image_height": pipeline_result.image.size[1],
            "image_width": pipeline_result.image.size[0],
            "gcps": [],
            LEVERS_OUTPUT_KEY: [],
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
            res.data[LEVERS_OUTPUT_KEY].append(p)
        return res


class GeoreferencingOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        # capture query points as output
        query_points = pipeline_result.data[QUERY_POINTS_OUTPUT_KEY]
        crs = pipeline_result.data[CRS_OUTPUT_KEY]
        confidence = 0

        # create ground control points from query point output data
        gcps = [
            LARAGroundControlPoint(
                id=f"gcp.{i}",
                pixel_x=qp.xy[0],
                pixel_y=qp.xy[1],
                latitude=qp.lonlat[1],
                longitude=qp.lonlat[0],
                confidence=qp.confidence,
            )
            for i, qp in enumerate(query_points)
        ]

        # create the final georeference result
        result = GeoreferenceResult(
            map_id=pipeline_result.raster_id,
            gcps=gcps,
            crs=crs,
            provenance="modelled",
            confidence=confidence,
        )

        result.gcps = gcps
        return BaseModelOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, result
        )


class CDROutput(GeoreferencingOutput):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        assert pipeline_result.image is not None

        results = super().create_output(pipeline_result)
        if not isinstance(results, BaseModelOutput) or not isinstance(
            results.data, GeoreferenceResult
        ):
            logger.error("Failed to create georeferencing cdr output")
            return results

        mapper = GeoreferenceMapper("georeferencing", "0.0.1")
        cdr_georeferencing = mapper.map_to_cdr(results.data)
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            cdr_georeferencing,
        )


class ProjectedMapOutput(OutputCreator):

    DEFAULT_OUTPUT_CRS = "EPSG:3857"

    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a projected GeoTIFF from the source map and tranformation / gcps computed
            by the georeferencing pipeline.
        Args:
            pipeline_result (PipelineResult): The result for the georeferencing pipeline.
        Returns:
            Output: An ImageOutput object wrapping the projected map.
        Raises:
            ValueError: If no image is found in the pipeline result.
        """

        # get the ground control points and source CRS
        crs = pipeline_result.data[CRS_OUTPUT_KEY]
        query_points: List[QueryPoint] = pipeline_result.data.get(
            QUERY_POINTS_OUTPUT_KEY, []
        )

        # create ground control points from query point output data
        gcps = [
            LARAGroundControlPoint(
                id=f"gcp.{i}",
                pixel_x=qp.xy[0],
                pixel_y=qp.xy[1],
                latitude=qp.lonlat[1],
                longitude=qp.lonlat[0],
                confidence=qp.confidence,
            )
            for i, qp in enumerate(query_points)
        ]

        # create the affine transformation matrix from the gcps
        transform = cps_to_transform(gcps, crs, ProjectedMapOutput.DEFAULT_OUTPUT_CRS)

        if pipeline_result.image is None:
            raise ValueError("No image found in pipeline result - cannot project")

        # project the image using the tansformation matrix - results are returned
        # as a geotiff in memory (pillow doesn't support geotiffs)
        projected_map = project_image(
            pipeline_result.image, transform, ProjectedMapOutput.DEFAULT_OUTPUT_CRS
        )

        return BytesOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, projected_map
        )
