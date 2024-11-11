import logging
import math

from geopy.distance import geodesic

from tasks.geo_referencing.entities import (
    CRS_OUTPUT_KEY,
    ERROR_SCALE_OUTPUT_KEY,
    KEYPOINTS_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    RMSE_OUTPUT_KEY,
    GroundControlPoint,
)
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
    MapROI,
    ROI_MAP_OUTPUT_KEY,
)
from tasks.geo_referencing.geo_projection import GeoProjection
from tasks.geo_referencing.util import get_input_geofence
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.metadata_extraction.scale import SCALE_VALUE_OUTPUT_KEY

from typing import Any, Dict, List, Optional, Tuple

import rasterio.transform as riot
from pyproj import Transformer


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


class PixelMapping:
    pixel_coord: float
    geo_coord: float

    def __init__(self, pixel_coord, geo_coord):
        self.pixel_coord = pixel_coord
        self.geo_coord = geo_coord


class GeoReference(Task):
    _poly_order = 1

    # externally supplied query points will be transformed to this CRS (NAD83 for contest data)
    EXTERNAL_QUERY_POINT_CRS = "EPSG:4269"

    def __init__(self, task_id: str, poly_order: int = 1):
        super().__init__(task_id)
        self._poly_order = poly_order

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
        return self._run_label_georef(input)

    def _run_label_georef(self, input: TaskInput) -> TaskResult:

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

        scale_value = input.get_data(SCALE_VALUE_OUTPUT_KEY)
        im_resize_ratio = input.get_data("im_resize_ratio", 1)

        roi_xy = []
        if ROI_MAP_OUTPUT_KEY in input.data:
            # get map ROI bounds (without inner/outer buffering)
            map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])
            roi_xy = map_roi.map_bounds

        roi_xy_minmax: Tuple[List[float], List[float]] = ([], [])
        if roi_xy is not None:
            roi_xy_minmax = (
                [min(map(lambda x: x[0], roi_xy)), max(map(lambda x: x[0], roi_xy))],
                [min(map(lambda x: x[1], roi_xy)), max(map(lambda x: x[1], roi_xy))],
            )
        else:
            # since no roi, use whole image as minmax values
            roi_xy_minmax = ([0, input.image.size[1]], [0, input.image.size[0]])

        if not scale_value:
            scale_value = 0

        query_pts = None
        external_query_pts = False
        if QUERY_POINTS_OUTPUT_KEY in input.request:
            logger.debug("reading query points from request")
            query_pts = input.request[QUERY_POINTS_OUTPUT_KEY]
            external_query_pts = True
        if not query_pts or len(query_pts) < 1:
            logger.debug("reading query points from task input")
            query_pts = input.get_data(QUERY_POINTS_OUTPUT_KEY)

        # if no clue point provided, build projections with fallbacks
        clue_point = input.get_request_info("clue_point")
        if clue_point is None:
            lon_check = list(map(lambda x: x[0], lon_pts))
            lat_check = list(map(lambda x: x[0], lat_pts))
            num_keypoints = min(len(lon_pts), len(lat_pts))

            lon_minmax_geofence = lon_minmax
            lat_minmax_geofence = lat_minmax
            if geofence is not None:
                lon_minmax_geofence, lat_minmax_geofence = self._get_fallback_geofence(
                    geofence
                )
                logger.debug(
                    f"adjusting geofence to lat: {lat_minmax_geofence} and lon: {lon_minmax_geofence}"
                )

            logger.debug(f"{num_keypoints} key points available for project")
            if len(lat_check) < 2 or (abs(max(lat_check) - min(lat_check)) > 20):
                anchors = self._build_fallback(roi_xy_minmax[1], lat_minmax_geofence)

                # create the anchor coordinates using the x mid range for the pixel coordinate
                lat_pts.clear()
                for a in anchors:
                    coord = Coordinate(
                        "lat keypoint",
                        f"fallback {a.geo_coord}",
                        a.geo_coord,
                        "anchor",
                        True,
                        pixel_alignment=(
                            (roi_xy_minmax[0][0] + roi_xy_minmax[0][1]) / 2,
                            a.pixel_coord,
                        ),
                        confidence=0.0,
                    )
                    _, y_pixel = coord.get_pixel_alignment()
                    lat_pts[(a.geo_coord, y_pixel)] = coord

            if len(lon_check) < 2 or (abs(max(lon_check) - min(lon_check)) > 20):
                anchors = self._build_fallback(roi_xy_minmax[0], lon_minmax_geofence)

                # create the anchor coordinates using the y mid range for the pixel coordinate
                lon_pts.clear()
                for a in anchors:
                    coord = Coordinate(
                        "lon keypoint",
                        f"fallback {a.geo_coord}",
                        a.geo_coord,
                        "anchor",
                        False,
                        pixel_alignment=(
                            a.pixel_coord,
                            (roi_xy_minmax[1][0] + roi_xy_minmax[1][1]) / 2,
                        ),
                        confidence=0.0,
                    )
                    x_pixel, _ = coord.get_pixel_alignment()
                    lon_pts[(a.geo_coord, x_pixel)] = coord

        confidence = 0
        lon_check = list(map(lambda x: x[0], lon_pts))
        lat_check = list(map(lambda x: x[0], lat_pts))
        num_keypoints = min(len(lon_pts), len(lat_pts))
        keypoint_stats = {}
        if (
            num_keypoints < 2
            or (abs(max(lon_check) - min(lon_check)) > 20)
            or (abs(max(lat_check) - min(lat_check)) > 20)
        ):
            # still not enough key-points, just use 'clue' lon/lat as fallback query response
            logger.info("not enough key points to generate a projection")
            geo_projn = None
        else:
            # ----- Use extracted keypoints to estimate geographic projection
            logger.info("sufficient key points to generate a projection")
            confidence = self._calculate_confidence(lon_pts, lat_pts)
            logger.info(f"confidence of projection is {confidence}")
            geo_projn = GeoProjection(self._poly_order)
            geo_projn.estimate_pxl2geo_mapping(
                list(map(lambda x: x[1], lon_pts.items())),
                list(map(lambda x: x[1], lat_pts.items())),
                input.image.size,
            )
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
        lon_multiplier, lat_multiplier = self._determine_hemispheres(input, query_pts)
        logger.info(
            f"derived hemispheres for georeferencing: {lon_multiplier},{lat_multiplier}"
        )
        # results = self._clip_query_pts(query_pts, lon_minmax, lat_minmax)
        results = self._update_hemispheres(query_pts, lon_multiplier, lat_multiplier)

        crs = self._determine_crs(input)
        logger.info(f"extracted CRS: {crs}")

        # perform a datum shift if we're using externally supplied
        # query points (ie. eval scenario where we are passed query points from a file)
        if crs != self.EXTERNAL_QUERY_POINT_CRS and external_query_pts:
            logger.info(
                f"performing datum shift from {crs} to {self.EXTERNAL_QUERY_POINT_CRS}"
            )
            for qp in query_pts:
                proj = Transformer.from_crs(
                    crs, self.EXTERNAL_QUERY_POINT_CRS, always_xy=True
                )
                x_p, y_p = proj.transform(qp.lonlat[0], qp.lonlat[1])
                qp.lonlat = (x_p, y_p)

        rmse, scale_error = self._score_query_points(query_pts, scale_value)

        result = super()._create_result(input)
        result.output[QUERY_POINTS_OUTPUT_KEY] = results
        result.output[RMSE_OUTPUT_KEY] = rmse
        result.output[ERROR_SCALE_OUTPUT_KEY] = scale_error
        result.output[CRS_OUTPUT_KEY] = (
            self.EXTERNAL_QUERY_POINT_CRS if external_query_pts else crs
        )
        result.output[KEYPOINTS_OUTPUT_KEY] = keypoint_stats
        return result

    def _count_keypoints(
        self, points: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[str, int]:
        counts = {}
        for _, c in points.items():
            source = c.get_source()
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

    def _max_range(self, minmax: List[float], max_range, invert: bool) -> List[float]:
        if abs(minmax[1] - minmax[0]) <= max_range:
            return minmax

        mid_point = (minmax[1] + minmax[0]) / 2
        adjustment = max_range / 2

        if invert and mid_point > 0:
            # return the inverse minmax
            return [mid_point + adjustment, mid_point - adjustment]

        return [mid_point - adjustment, mid_point + adjustment]

    def _get_fallback_geofence(
        self, geofence: DocGeoFence
    ) -> Tuple[List[float], List[float]]:
        # adjust based on maximum range derived from scale
        # TODO: use scale to derive maximum range
        max_range = 0.5
        return self._max_range(
            geofence.geofence.lon_minmax, max_range, False
        ), self._max_range(geofence.geofence.lat_minmax, max_range, True)

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
        if (
            geofence is not None
            and not geofence.geofence.region_type == GeoFenceType.DEFAULT
        ):
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
                logger.debug(f"hemispheres derived entirely from geofence")
                return lon_multiplier, lat_multiplier

        # function assumes that north is up and that the image is not skewed
        # set east - west hemisphere by seeing how longitude changes when x increases
        if not lon_determined:
            qps_sorted_x = sorted(query_pts, key=lambda x: x.xy[0])
            if qps_sorted_x[0].lonlat[0] < 0:
                lon_multiplier = -1
            elif qps_sorted_x[0].lonlat[0] > qps_sorted_x[-1].lonlat[0]:
                # x increased but lon decreased so it is in the negative hemisphere
                lon_multiplier = -1  # if qps_sorted_x[0].lonlat[0] > 0 else 1
            logger.debug("longitude hemisphere in part determined by points")

        # set north - south hemisphere by seeing how latitude changes when y increases
        if not lat_determined:
            qps_sorted_y = sorted(query_pts, key=lambda x: x.xy[1])
            if qps_sorted_y[0].lonlat[1] < 0:
                lat_multiplier = -1
            elif abs(qps_sorted_y[0].lonlat[1]) < abs(qps_sorted_y[-1].lonlat[1]):
                # y increased and lat increased so it is in the negative hemisphere
                lat_multiplier = -1  # if qps_sorted_y[0].lonlat[1] > 0 else 1
            logger.debug("latitude hemisphere in part determined by points")

        return lon_multiplier, lat_multiplier

    def _update_hemispheres(
        self, query_pts: List[QueryPoint], lon_multiplier: float, lat_multiplier: float
    ) -> List[QueryPoint]:
        for qp in query_pts:
            qp.lonlat = (
                abs(qp.lonlat[0]) * lon_multiplier,
                abs(qp.lonlat[1]) * lat_multiplier,
            )
            if "lonlat_xp" in qp.properties:
                qp.lonlat_xp = (
                    abs(qp.lonlat_xp[0]) * lon_multiplier,
                    abs(qp.lonlat_xp[1]) * lat_multiplier,
                )
            if "lonlat_yp" in qp.properties:
                qp.lonlat_yp = (
                    abs(qp.lonlat_yp[0]) * lon_multiplier,
                    abs(qp.lonlat_yp[1]) * lat_multiplier,
                )

        return query_pts

    def _update_hemispheres_corners(
        self,
        corner_points: List[GroundControlPoint],
        lon_multiplier: float,
        lat_multiplier: float,
    ) -> List[GroundControlPoint]:
        for cp in corner_points:
            cp.longitude = abs(cp.longitude) * lon_multiplier
            cp.latitude = abs(cp.latitude) * lat_multiplier

        return corner_points

    def _determine_crs(self, input: TaskInput) -> str:
        # parse extracted metadata
        metadata: Optional[MetadataExtraction] = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )

        # make sure there is metadata
        if not metadata:
            return self.EXTERNAL_QUERY_POINT_CRS

        # grab the year from the metadata if present
        try:
            year = int(metadata.year)
        except:
            year = -1

        # we we assume geographic coordinates and combine that with the datum to
        # come up with a CRS
        datum = metadata.datum.upper()
        if datum is not None and datum != "NULL":
            if "NAD" in datum or "NORTH AMERICAN" in datum:
                if "27" or "1927" in datum:
                    return "EPSG:4267"
                if "83" or "1983" in datum:
                    return "EPSG:4269"
                if year >= 1985:
                    return "EPSG:4269"
                if year >= 1930:
                    return "EPSG:4267"
            # default to a WGS84 CRS
            return "EPSG:4326"

        # no datum info in the metadata so we will use the country and year
        if not datum or datum == "NULL" or len(datum) == 0:
            if metadata.country != "NULL" and (
                metadata.country == "US" or metadata.country == "CA"
            ):
                if year >= 1985:
                    return "EPSG:4269"
                if year >= 1930:
                    return "EPSG:4267"

        # default to a WGS84 CRS when all else fails
        return "EPSG:4326"

    def _build_fallback(
        self,
        coord_minmax: List[float],
        geo_minmax: List[float],
    ) -> List[PixelMapping]:
        logger.debug(
            f"building fallback anchors mapping {coord_minmax} and {geo_minmax}"
        )
        # calculate the adjustment amount to shift the anchors inside the range
        pixel_adjustment = (
            coord_minmax[1] - coord_minmax[0]
        ) * FALLBACK_RANGE_ADJUSTMENT_FACTOR
        geo_adjustment = (
            geo_minmax[1] - geo_minmax[0]
        ) * FALLBACK_RANGE_ADJUSTMENT_FACTOR

        # assume the pixel range matches the geo range, anchor the match near the end of the range
        coord_anchors = [
            coord_minmax[0] + pixel_adjustment,
            coord_minmax[1] - pixel_adjustment,
        ]
        geo_anchors = [geo_minmax[0] + geo_adjustment, geo_minmax[1] - geo_adjustment]

        # build the mapping for min, mid, max
        return [
            PixelMapping(pixel_coord=coord_anchors[0], geo_coord=geo_anchors[0]),
            PixelMapping(
                pixel_coord=(coord_anchors[0] + coord_anchors[1]) / 2,
                geo_coord=(geo_anchors[0] + geo_anchors[1]) / 2,
            ),
            PixelMapping(pixel_coord=coord_anchors[1], geo_coord=geo_anchors[1]),
        ]

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

    def _clip(self, value: float, lower: float, upper: float) -> float:
        return lower if value < lower else upper if value > upper else value

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
