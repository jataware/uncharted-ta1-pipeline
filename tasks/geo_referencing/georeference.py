import logging
import math
from pyproj import CRS as pyproj_CRS
from geopy.distance import geodesic

from tasks.geo_referencing.entities import (
    CRS_OUTPUT_KEY,
    ERROR_SCALE_OUTPUT_KEY,
    KEYPOINTS_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    RMSE_OUTPUT_KEY,
)
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
)
from tasks.geo_referencing.geo_projection import GeoProjection
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from typing import Any, Dict, List, Optional, Tuple
from pyproj import Transformer


EXTERNAL_QUERY_POINT_CRS = "EPSG:4269"  # externally supplied query points will be transformed to this CRS (NAD83 for contest data)
FALLBACK_RANGE_ADJUSTMENT_FACTOR = 0.05  # used to calculate how far from the edge of the fallback range to anchor points

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
    error_scale: Optional[float]

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

        self.error_km: Optional[float] = None
        self.error_scale: Optional[float] = None
        self.dist_xp_km: Optional[float] = None
        self.dist_yp_km: Optional[float] = None

        self.confidence = confidence


class GeoReference(Task):

    def __init__(
        self,
        task_id: str,
        poly_order: int = 1,
        external_crs: str = EXTERNAL_QUERY_POINT_CRS,
    ):
        super().__init__(task_id)
        self._poly_order = poly_order
        self._external_crs = external_crs

    def run(self, input: TaskInput) -> TaskResult:
        """
        Creates a transformation from pixel coordinates to geographic coordinates
        based on exracted map information.  The transformation is run against supplied
        query points in pixel space to generate a final set of geographic coordinates.

        These pixel/geo point tuples can be used downstream as control points.

        Args:
            input (TaskInput): The input data for the georeferencing task.

        Returns:
            TaskResult: The result of the georeferencing task.
        """

        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        lon_minmax = input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.get_request_info("lat_minmax", [0, 90])
        if geofence:
            lon_minmax = geofence.geofence.lon_minmax
            lat_minmax = geofence.geofence.lat_minmax

        lon_pts = input.get_data("lons", {})
        lat_pts = input.get_data("lats", {})

        # TODO -- this scale_value has always been 0, due to the old ScaleExtractor being disabled
        # try using the new scale analysis result here
        scale_value = 0
        im_resize_ratio = input.get_data("im_resize_ratio", 1)

        query_pts = None
        external_query_pts = False
        if QUERY_POINTS_OUTPUT_KEY in input.request:
            logger.debug("reading query points from request")
            query_pts = input.request[QUERY_POINTS_OUTPUT_KEY]
            external_query_pts = True
        if not query_pts or len(query_pts) < 1:
            logger.debug("reading query points from task input")
            query_pts = input.get_data(QUERY_POINTS_OUTPUT_KEY)

        confidence = 0
        num_keypoints = min(len(lon_pts), len(lat_pts))
        keypoint_stats = {}
        geo_projn: Optional[GeoProjection] = None
        if num_keypoints < 2:
            # still not enough key-points, just use 'clue' lon/lat as fallback query response
            logger.warning("Not enough keypoints to generate a geoprojection")
            geo_projn = None
        else:
            # ----- Use extracted keypoints to estimate geographic projection
            logger.info("Generating geoprojection transform...")
            confidence = self._calculate_confidence(lon_pts, lat_pts)
            logger.info(f"confidence of projection is {confidence}")
            geo_projn = GeoProjection(self._poly_order)
            projn_ok = geo_projn.estimate_pxl2geo_mapping(
                list(map(lambda x: x[1], lon_pts.items())),
                list(map(lambda x: x[1], lat_pts.items())),
            )
            if not projn_ok:
                logger.warning("geoprojection calc was unsuccessful! Forcing to None")
                geo_projn = None
            keypoint_stats["lats"] = self._count_keypoints(lat_pts)
            keypoint_stats["lons"] = self._count_keypoints(lon_pts)

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
        lonlat_multiplier = self._get_hemisphere_multiplier(input, query_pts)
        results = self._apply_hemisphere_multiplier(query_pts, lonlat_multiplier)

        source_crs = self._determine_crs(input)
        logger.info(f"extracted CRS: {source_crs}")

        # perform a datum shift if we're using externally supplied
        # query points (ie. eval scenario where we are passed query points from a file)
        if source_crs != self._external_crs and external_query_pts:
            logger.info(
                f"performing datum shift from {source_crs} to {self._external_crs}"
            )
            for qp in query_pts:
                proj = Transformer.from_crs(
                    source_crs, self._external_crs, always_xy=True
                )
                x_p, y_p = proj.transform(qp.lonlat[0], qp.lonlat[1])
                qp.lonlat = (x_p, y_p)

        rmse, scale_error = self._score_query_points(query_pts, scale_value)

        result = super()._create_result(input)
        result.output[QUERY_POINTS_OUTPUT_KEY] = results
        result.output[RMSE_OUTPUT_KEY] = rmse
        result.output[ERROR_SCALE_OUTPUT_KEY] = scale_error
        result.output[CRS_OUTPUT_KEY] = (
            self._external_crs if external_query_pts else source_crs
        )
        result.output[KEYPOINTS_OUTPUT_KEY] = keypoint_stats
        return result

    def _count_keypoints(
        self, points: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[str, int]:
        counts = {}
        for _, c in points.items():
            source = str(c.get_source().value)
            if source not in counts:
                counts[source] = 0
            counts[source] = counts[source] + 1
        return counts

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
            logger.error(f"EXCEPTION geo projecting: {repr(e)}")
            return self._add_fallback(input, query_pts, lon_minmax, lat_minmax)

        return results

    def _get_hemisphere_multiplier(
        self, input: TaskInput, query_pts: List[QueryPoint]
    ) -> Tuple[float, float]:
        """
        Returns +1 for Northern latitudes or Eastern longitudes,
        returns -1 for Southern latitudes or Western longitudes
        """
        # use the geofence if it was not defaulted and indicates a clear hemisphere
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        if (
            geofence is not None
            and not geofence.geofence.region_type == GeoFenceType.DEFAULT
        ):
            # hemispheres were already determined by geofence task
            return geofence.geofence.lonlat_hemispheres

        # assume that north is up and that the image is not skewed
        # set east - west hemisphere by seeing how longitude changes when x increases
        lon_multiplier = 1
        qps_sorted_x = sorted(query_pts, key=lambda x: x.xy[0])
        if qps_sorted_x[0].lonlat[0] < 0:
            lon_multiplier = -1
        elif qps_sorted_x[0].lonlat[0] > qps_sorted_x[-1].lonlat[0]:
            # x increased but lon decreased so it is in the negative hemisphere
            lon_multiplier = -1
        logger.debug("longitude hemisphere in part determined by points")

        # set north - south hemisphere by seeing how latitude changes when y increases
        lat_multiplier = 1
        qps_sorted_y = sorted(query_pts, key=lambda x: x.xy[1])
        if qps_sorted_y[0].lonlat[1] < 0:
            lat_multiplier = -1
        elif abs(qps_sorted_y[0].lonlat[1]) < abs(qps_sorted_y[-1].lonlat[1]):
            # y increased and lat increased so it is in the negative hemisphere
            lat_multiplier = -1
        logger.debug("latitude hemisphere in part determined by points")

        return lon_multiplier, lat_multiplier

    def _apply_hemisphere_multiplier(
        self, query_pts: List[QueryPoint], lonlat_multiplier: Tuple[float, float]
    ) -> List[QueryPoint]:
        """
        Apply hemisphere sign multipliers to lon/lat query pt results
        """
        for qp in query_pts:
            qp.lonlat = (
                abs(qp.lonlat[0]) * lonlat_multiplier[0],
                abs(qp.lonlat[1]) * lonlat_multiplier[1],
            )
            if "lonlat_xp" in qp.properties:
                qp.lonlat_xp = (
                    abs(qp.lonlat_xp[0]) * lonlat_multiplier[0],
                    abs(qp.lonlat_xp[1]) * lonlat_multiplier[1],
                )
            if "lonlat_yp" in qp.properties:
                qp.lonlat_yp = (
                    abs(qp.lonlat_yp[0]) * lonlat_multiplier[0],
                    abs(qp.lonlat_yp[1]) * lonlat_multiplier[1],
                )

        return query_pts

    def _determine_crs(self, input: TaskInput) -> str:
        # parse extracted metadata
        metadata: Optional[MetadataExtraction] = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )

        # make sure there is metadata
        if not metadata:
            return self._external_crs

        # get the computed CRS from the metadata if it exists
        crs = metadata.crs
        if crs is not None and crs != "NULL":
            try:
                # strip "EPSG:" from the CRS if present
                crs_num = int(crs.replace("EPSG:", ""))

                # get the geographic CRS info
                pcrs = pyproj_CRS.from_epsg(crs_num)
                if pcrs.is_projected:
                    gcrs = pcrs.geodetic_crs
                    if gcrs:
                        crs_num = gcrs.to_epsg()
                return str(f"EPSG:{crs_num}")
            except:
                logger.exception(
                    f"failed to extract geographic CRS from metadata CRS: {crs}"
                )
                crs = "NULL"

        # no crs info in the metadata so we will use the country and year
        if not crs or crs == "NULL" or len(crs) == 0:
            # grab the year from the metadata if present
            try:
                year = int(metadata.year)
            except:
                year = -1

            if metadata.country != "NULL" and (
                metadata.country == "US" or metadata.country == "CA"
            ):
                if year >= 1990:
                    return "EPSG:4269"
                if year >= 1930:
                    return "EPSG:4267"

        # default to a WGS84 CRS when all else fails
        return "EPSG:4326"

    def _add_fallback(
        self,
        input: TaskInput,
        query_pts: List[QueryPoint],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> List[QueryPoint]:
        logger.debug(
            f"adding fallback when georeferencing using {lon_minmax} & {lat_minmax}"
        )
        results = []

        # use the geofence if it was not defaulted and lat/lon minmax not clue point based
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        if (
            geofence is not None
            and not geofence.geofence.region_type == GeoFenceType.DEFAULT
        ):
            lon_minmax = geofence.geofence.lon_minmax
            lat_minmax = geofence.geofence.lat_minmax
            logger.debug(
                f"adjusting fallback window to geofence of {lon_minmax} & {lat_minmax}"
            )

        # no geographic-projection polynomial available,
        # just use the 'clue' midpoint as a fallback answer for any query points
        lon_clue = abs(
            (lon_minmax[0] + lon_minmax[1]) / 2
        )  # mid-points of lon/lat clue area
        lat_clue = abs((lat_minmax[0] + lat_minmax[1]) / 2)
        for qp in query_pts:
            qp.lonlat = (lon_clue, lat_clue)
            qp.lonlat_xp = (lon_clue, lat_clue)
            qp.lonlat_yp = (lon_clue, lat_clue)
            qp.confidence = 0
            results.append(qp)
        return results

    def _score_query_points(
        self, query_pts: List[QueryPoint], scale_value: float
    ) -> Tuple[float, float]:
        # if ground truth lon/lat info exists, calculate the
        # RMSE of geodesic error distances (in km) for all
        # query points for a given image

        sum_sq_error = 0.0
        sum_sq_scale_error = 0.0
        num_pts = 0
        for qp in query_pts:
            if qp.lonlat_gtruth is not None:
                latlon_gtruth = (qp.lonlat_gtruth[1], qp.lonlat_gtruth[0])
                latlon = (qp.lonlat[1], qp.lonlat[0])
                err_dist = geodesic(latlon_gtruth, latlon).km
                qp.error_km = err_dist
                if "lonlat_xp" in qp.properties:
                    qp.dist_xp_km = geodesic(
                        latlon, (qp.lonlat_xp[1], qp.lonlat_xp[0])
                    ).km
                if "lonlat_yp" in qp.properties:
                    qp.dist_yp_km = geodesic(
                        latlon, (qp.lonlat_yp[1], qp.lonlat_yp[0])
                    ).km
                qp.error_lonlat = (
                    latlon[1] - latlon_gtruth[1],
                    latlon[0] - latlon_gtruth[0],
                )
                if scale_value > 0 and qp.error_km is not None:
                    # calculate the error based on heuristic of scale / 1000 (in meters)
                    qp.error_scale = (qp.error_km * 1000.0) / (
                        float(scale_value) / 1000.0
                    )
                    sum_sq_scale_error += qp.error_scale * qp.error_scale

                sum_sq_error += err_dist * err_dist
                num_pts += 1

        if num_pts == 0:
            return -1, -1
        rmse = math.sqrt(sum_sq_error / num_pts)

        # all points in an image will either have a scale error or not have one so dont need to track point count separately
        scale_error = -1
        if scale_value > 0:
            scale_error = math.sqrt(sum_sq_scale_error / num_pts)
        logger.info(f"rmse: {rmse} scale error: {scale_error}")

        return rmse, scale_error
