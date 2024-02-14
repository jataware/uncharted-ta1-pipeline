import logging
import math

from geopy.distance import geodesic

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import Coordinate, DocGeoFence, GEOFENCE_OUTPUT_KEY
from tasks.geo_referencing.geo_projection import GeoProjection
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("geo_referencing")


class QueryPoint:
    id: str
    xy: Tuple[float, float]
    lonlat_gtruth: Optional[Tuple[float, float]]
    properties: Dict[str, Any]
    lonlat: Tuple[float, float]
    lonlat_xp: Tuple[float, float]
    lonlat_yp: Tuple[float, float]
    error_lonlat: Tuple[float, float]
    confidence: float

    def __init__(
        self,
        id: str,
        xy: Tuple[float, float],
        lonlat_gtruth: Optional[Tuple[float, float]],
        properties: Dict[str, Any] = {},
        confidence: float = 0,
    ):
        self.id = id
        self.xy = xy
        self.lonlat_gtruth = lonlat_gtruth
        self.properties = properties

        self.error_km = None
        self.dist_xp_km = None
        self.dist_yp_km = None

        self.confidence = confidence


class GeoReference(Task):
    _poly_order = 1

    def __init__(self, task_id: str, poly_order: int = 1):
        super().__init__(task_id)
        self._poly_order = poly_order

    def run(self, input: TaskInput) -> TaskResult:
        lon_minmax = input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.get_request_info("lat_minmax", [0, 90])
        logger.info(f"initial lon_minmax: {lon_minmax}")
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")
        im_resize_ratio = input.get_data("im_resize_ratio", 1)

        query_pts = None
        if "query_pts" in input.request:
            query_pts = input.request["query_pts"]
        if not query_pts or len(query_pts) < 1:
            query_pts = input.get_data("query_pts")

        lon_check = list(map(lambda x: x[0], lon_pts))
        lat_check = list(map(lambda x: x[0], lat_pts))
        print(f"LAT CHECK: {lat_check}")
        print(f"LON CHECK: {lon_check}")
        num_keypoints = min(len(lon_pts), len(lat_pts))
        confidence = 0
        if (
            num_keypoints < 2
            or (abs(max(lon_check) - min(lon_check)) > 40)
            or (abs(max(lat_check) - min(lat_check)) > 40)
        ):
            # still not enough key-points, just use 'clue' lon/lat as fallback query response
            geo_projn = None
        else:
            # ----- Use extracted keypoints to estimate geographic projection
            confidence = self._calculate_confidence(lon_pts, lat_pts)
            geo_projn = GeoProjection(self._poly_order)
            geo_projn.estimate_pxl2geo_mapping(
                list(map(lambda x: x[1], lon_pts.items())),
                list(map(lambda x: x[1], lat_pts.items())),
                input.image.size,
            )

        # ----- Get lon/lat results for query points for this image
        results = self._process_query_points(
            input,
            query_pts,
            im_resize_ratio,
            geo_projn,
            lon_minmax,
            lat_minmax,
            confidence,
        )
        lon_multiplier, lat_multiplier = self._determine_hemispheres(input, query_pts)
        results = self._clip_query_pts(query_pts, lon_minmax, lat_minmax)
        results = self._update_hemispheres(query_pts, lon_multiplier, lat_multiplier)

        rmse = self._score_query_points(query_pts)
        datum, projection = self._determine_projection(input)
        logger.info(f"extracted datum: {datum}\textracted projection: {projection}")

        result = super()._create_result(input)
        result.output["query_pts"] = results
        result.output["rmse"] = rmse
        result.output["datum"] = datum
        result.output["projection"] = projection
        return result

    def _calculate_confidence(
        self,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> float:
        # start with lon confidence
        lon_count = 0
        lon_conf = 1.0
        for _, l in lons.items():
            lon_count = lon_count + 1
            lon_conf = lon_conf * l.get_confidence()

            # adjust for when more points are provided
            if lon_count > 2:
                lon_conf = lon_conf * 1.1

        # do the same for lat
        lat_count = 0
        lat_conf = 1.0
        for _, l in lats.items():
            lat_count = lat_count + 1
            lat_conf = lat_conf * l.get_confidence()

            # adjust for when more points are provided
            if lat_count > 2:
                lat_conf = lat_conf * 1.1

        # combine them
        return min(1, lon_conf * lat_conf)

    def _process_query_points(
        self,
        input: TaskInput,
        query_pts: List[QueryPoint],
        im_resize_ratio: float,
        geo_projn: Optional[GeoProjection],
        lon_minmax: List[float],
        lat_minmax: List[float],
        confidence: float,
    ) -> List[QueryPoint]:
        if geo_projn is None:
            return self._add_fallback(input, query_pts, lon_minmax, lat_minmax)

        # use geographic-projection polynomial to estimate lon/lat for query points
        results = []
        try:
            xy_queries = (
                [
                    (qp.xy[0] * im_resize_ratio, qp.xy[1] * im_resize_ratio)
                    for qp in query_pts
                ],
                [
                    (qp.xy[0] * im_resize_ratio + 1, qp.xy[1] * im_resize_ratio)
                    for qp in query_pts
                ],
                [
                    (qp.xy[0] * im_resize_ratio, qp.xy[1] * im_resize_ratio + 1)
                    for qp in query_pts
                ],
            )
            lonlat_estimates = (
                geo_projn.predict_xy_pts(xy_queries[0]),
                geo_projn.predict_xy_pts(xy_queries[1]),
                geo_projn.predict_xy_pts(xy_queries[2]),
            )
            for lonlat, lonlat_xp, lonlat_yp, qp in zip(
                lonlat_estimates[0], lonlat_estimates[1], lonlat_estimates[2], query_pts
            ):
                qp.lonlat = lonlat
                qp.lonlat_xp = lonlat_xp
                qp.lonlat_yp = lonlat_yp
                qp.confidence = confidence
                results.append(qp)
        except Exception as e:
            logger.error(f"EXCEPTION geo projecting")
            logger.error(e)
            return self._add_fallback(input, query_pts, lon_minmax, lat_minmax)

        return results

    def _clip_query_pts(
        self,
        query_pts: List[QueryPoint],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> List[QueryPoint]:
        if lon_minmax and lat_minmax:
            for qp in query_pts:
                # ensure query pt results are within expected field-of-view range
                qp.lonlat = (
                    self._clip(qp.lonlat[0], lon_minmax[0], lon_minmax[1]),
                    self._clip(qp.lonlat[1], lat_minmax[0], lat_minmax[1]),
                )
                qp.lonlat_xp = (
                    self._clip(qp.lonlat_xp[0], lon_minmax[0], lon_minmax[1]),
                    self._clip(qp.lonlat_xp[1], lat_minmax[0], lat_minmax[1]),
                )
                qp.lonlat_yp = (
                    self._clip(qp.lonlat_yp[0], lon_minmax[0], lon_minmax[1]),
                    self._clip(qp.lonlat_yp[1], lat_minmax[0], lat_minmax[1]),
                )
        return query_pts

    def _determine_hemispheres(
        self, input: TaskInput, query_pts: List[QueryPoint]
    ) -> Tuple[float, float]:
        lon_multiplier = 1
        lon_determined = False
        lat_multiplier = 1
        lat_determined = False

        # use the geofence if it was not defaulted and indicates a clear hemisphere
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        if geofence is not None and not geofence.geofence.defaulted:
            # check if longitude min and max are both in the same hemisphere
            if (
                geofence.geofence.lon_minmax[0] <= 0
                and geofence.geofence.lon_minmax[1] <= 0
            ):
                lon_multiplier = -1
                lon_determined = True
            elif (
                geofence.geofence.lon_minmax[0] >= 0
                and geofence.geofence.lon_minmax[1] >= 0
            ):
                lon_determined = True

            # check if latitude min and max are both in the same hemisphere
            if (
                geofence.geofence.lat_minmax[0] <= 0
                and geofence.geofence.lat_minmax[1] <= 0
            ):
                lat_multiplier = -1
                lat_determined = True
            elif (
                geofence.geofence.lat_minmax[0] >= 0
                and geofence.geofence.lat_minmax[1] >= 0
            ):
                lat_determined = True

            if lat_determined and lon_determined:
                return lon_multiplier, lat_multiplier

        # function assumes that north is up and that the image is not skewed
        # set east - west hemisphere by seeing how longitude changes when x increases
        if not lon_determined:
            qps_sorted_x = sorted(query_pts, key=lambda x: x.xy[0])
            if abs(qps_sorted_x[0].lonlat[0]) > abs(qps_sorted_x[-1].lonlat[0]):
                # x increased but lon decreased so it is negative
                lon_multiplier = -1

        # set north - south hemisphere by seeing how latitude changes when y increases
        if not lat_determined:
            qps_sorted_y = sorted(query_pts, key=lambda x: x.xy[1])
            if abs(qps_sorted_y[0].lonlat[1]) < abs(qps_sorted_y[-1].lonlat[1]):
                # y increased and lat increased so it is negative
                lat_multiplier = -1

        return lon_multiplier, lat_multiplier

    def _update_hemispheres(
        self, query_pts: List[QueryPoint], lon_multiplier: float, lat_multiplier: float
    ) -> List[QueryPoint]:
        for qp in query_pts:
            qp.lonlat = (
                abs(qp.lonlat[0]) * lon_multiplier,
                abs(qp.lonlat[1]) * lat_multiplier,
            )
            qp.lonlat_xp = (
                abs(qp.lonlat_xp[0]) * lon_multiplier,
                abs(qp.lonlat_xp[1]) * lat_multiplier,
            )
            qp.lonlat_yp = (
                abs(qp.lonlat_yp[0]) * lon_multiplier,
                abs(qp.lonlat_yp[1]) * lat_multiplier,
            )

        return query_pts

    def _determine_projection(self, input: TaskInput) -> Tuple[str, str]:
        # parse extracted metadata
        metadata = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )

        # make sure there is metadata
        if not metadata:
            return "", ""

        datum = metadata.datum
        if not datum or len(datum) == 0:
            year = metadata.year
            if year >= "1985":
                datum = "NAD83"
            if year >= "1930":
                datum = "NAD27"

        # return the datum and the projection
        return datum, metadata.projection

    def _add_fallback(
        self,
        input: TaskInput,
        query_pts: List[QueryPoint],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> List[QueryPoint]:
        logger.info(
            f"adding fallback when georeferencing using {lon_minmax} & {lat_minmax}"
        )
        results = []

        # use the geofence if it was not defaulted and lat/lon minmax not clue point based
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        if geofence is not None and not geofence.geofence.defaulted:
            lon_minmax = geofence.geofence.lon_minmax
            lat_minmax = geofence.geofence.lat_minmax
            logger.info(
                f"adjusting fallback window to geofence of {lon_minmax} & {lat_minmax}"
            )

        # no geographic-projection polynomial available,
        # just use the 'clue' midpoint as a fallback answer for any query points
        lon_clue = (
            lon_minmax[0] + lon_minmax[1]
        ) / 2  # mid-points of lon/lat clue area
        lat_clue = (lat_minmax[0] + lat_minmax[1]) / 2
        for qp in query_pts:
            qp.lonlat = (lon_clue, lat_clue)
            qp.lonlat_xp = (lon_clue, lat_clue)
            qp.lonlat_yp = (lon_clue, lat_clue)
            qp.confidence = 0
            results.append(qp)
        return results

    def _clip(self, value: float, lower: float, upper: float) -> float:
        return lower if value < lower else upper if value > upper else value

    def _score_query_points(self, query_pts: List[QueryPoint]) -> float:
        logger.info("scoring georeferencing...")
        # if ground truth lon/lat info exists, calculate the
        # RMSE of geodesic error distances (in km) for all
        # query points for a given image

        sum_sq_error = 0.0
        num_pts = 0
        for qp in query_pts:
            if qp.lonlat_gtruth is not None:
                latlon_gtruth = (qp.lonlat_gtruth[1], qp.lonlat_gtruth[0])
                latlon = (qp.lonlat[1], qp.lonlat[0])
                err_dist = geodesic(latlon_gtruth, latlon).km
                qp.error_km = err_dist
                qp.dist_xp_km = geodesic(latlon, (qp.lonlat_xp[1], qp.lonlat_xp[0])).km
                qp.dist_yp_km = geodesic(latlon, (qp.lonlat_yp[1], qp.lonlat_yp[0])).km
                qp.error_lonlat = (
                    latlon[1] - latlon_gtruth[1],
                    latlon[0] - latlon_gtruth[0],
                )

                sum_sq_error += err_dist * err_dist
                num_pts += 1

        if num_pts == 0:
            return -1
        rmse = math.sqrt(sum_sq_error / num_pts)
        return rmse
