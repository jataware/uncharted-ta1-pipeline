import math

from geopy.distance import geodesic

from compute.geo_projection import GeoProjection
from tasks.geo_referencing.task import Task, TaskInput, TaskResult


class QueryPoint:
    def __init__(self, id, xy: tuple, lonlat_gtruth: tuple):
        self.id = id
        self.xy = xy
        self.lonlat_gtruth = lonlat_gtruth
        self.lonlat = None
        self.lonlat_xp = None
        self.lonlat_yp = None

        self.error_km = None
        self.dist_xp_km = None
        self.dist_yp_km = None
        self.error_lonlat = None


class GeoReference(Task):
    _poly_order = 1

    def __init__(self, task_id: str, poly_order: int = 1):
        super().__init__(task_id)
        self._poly_order = poly_order

    def run(self, input: TaskInput) -> TaskResult:
        lon_minmax = input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.get_request_info("lat_minmax", [0, 90])
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")
        lon_sign_factor = input.get_request_info("lon_sign_factor", 1)
        im_resize_ratio = input.get_data("im_resize_ratio", 1)

        query_pts = input.request["query_pts"]

        lon_check = list(map(lambda x: x[0], lon_pts))
        lat_check = list(map(lambda x: x[0], lat_pts))
        print(f"lon check: {lon_check}\tlat check: {lat_check}")
        print(f"lons: {lon_pts}")
        print(f"lats: {lat_pts}")
        num_keypoints = min(len(lon_pts), len(lat_pts))
        if (
            num_keypoints < 2
            or (abs(max(lon_check) - min(lon_check)) > 40)
            or (abs(max(lat_check) - min(lat_check)) > 40)
        ):
            # still not enough key-points, just use 'clue' lon/lat as fallback query response
            geo_projn = None
        else:
            # ----- Use extracted keypoints to estimate geographic projection
            geo_projn = GeoProjection(self._poly_order)
            geo_projn.estimate_pxl2geo_mapping(
                list(map(lambda x: x[1], lon_pts.items())),
                list(map(lambda x: x[1], lat_pts.items())),
                lon_sign_factor,
                input.image.size,
            )

        # ----- Get lon/lat results for query points for this image
        results = self._process_query_points(
            query_pts,
            im_resize_ratio,
            geo_projn,
            lon_minmax,
            lat_minmax,
            lon_sign_factor,
        )

        rmse = self._score_query_points(query_pts)

        result = super()._create_result(input)
        result.output["query_pts"] = results
        result.output["rmse"] = rmse
        return result

    def _process_query_points(
        self,
        query_pts: list,
        im_resize_ratio: float,
        geo_projn: GeoProjection,
        lon_minmax: tuple,
        lat_minmax: tuple,
        lon_sign_factor: float,
    ):
        if geo_projn is None:
            return self._add_fallback(
                query_pts, lon_minmax, lat_minmax, lon_sign_factor
            )

        # use geographic-projection polynomial to estimate lon/lat for query points
        results = []
        try:
            xy_queries = (
                [
                    [qp.xy[0] * im_resize_ratio, qp.xy[1] * im_resize_ratio]
                    for qp in query_pts
                ],
                [
                    [qp.xy[0] * im_resize_ratio + 1, qp.xy[1] * im_resize_ratio]
                    for qp in query_pts
                ],
                [
                    [qp.xy[0] * im_resize_ratio, qp.xy[1] * im_resize_ratio + 1]
                    for qp in query_pts
                ],
            )
            lonlat_estimates = (
                geo_projn.predict_xy_pts(xy_queries[0]),
                geo_projn.predict_xy_pts(xy_queries[1]),
                geo_projn.predict_xy_pts(xy_queries[2]),
            )

            if lon_minmax and lat_minmax:
                if lon_sign_factor < 0.0:
                    lon_minmax = (
                        lon_sign_factor * max(lon_minmax),
                        lon_sign_factor * min(lon_minmax),
                    )

                for lonlat, lonlat_xp, lonlat_yp, qp in zip(
                    lonlat_estimates[0],
                    lonlat_estimates[1],
                    lonlat_estimates[2],
                    query_pts,
                ):
                    # ensure query pt results are within expected field-of-view range
                    qp.lonlat = (
                        self._clip(lonlat[0], lon_minmax[0], lon_minmax[1]),
                        self._clip(lonlat[1], lat_minmax[0], lat_minmax[1]),
                    )
                    qp.lonlat_xp = (
                        self._clip(lonlat_xp[0], lon_minmax[0], lon_minmax[1]),
                        self._clip(lonlat_xp[1], lat_minmax[0], lat_minmax[1]),
                    )
                    qp.lonlat_yp = (
                        self._clip(lonlat_yp[0], lon_minmax[0], lon_minmax[1]),
                        self._clip(lonlat_yp[1], lat_minmax[0], lat_minmax[1]),
                    )
                    results.append(qp)
            else:
                for lonlat, lonlat_xp, lonlat_yp, qp in zip(
                    lonlat_estimates[0],
                    lonlat_estimates[1],
                    lonlat_estimates[2],
                    query_pts,
                ):
                    qp.lonlat = lonlat
                    qp.lonlat_xp = lonlat_xp
                    qp.lonlat_yp = lonlat_yp
                    results.append(qp)
        except Exception as e:
            print(f"EXCEPTION geo projecting")
            print(e)
            return self._add_fallback(
                query_pts, lon_minmax, lat_minmax, lon_sign_factor
            )

        return results

    def _add_fallback(
        self,
        query_pts: list,
        lon_minmax: tuple,
        lat_minmax: tuple,
        lon_sign_factor: float,
    ):
        print("adding fallback when georeferencing")
        results = []
        # no geographic-projection polynomial available,
        # just use the 'clue' midpoint as a fallback answer for any query points
        lon_clue = (
            lon_sign_factor * (lon_minmax[0] + lon_minmax[1]) / 2
        )  # mid-points of lon/lat clue area
        lat_clue = (lat_minmax[0] + lat_minmax[1]) / 2
        for qp in query_pts:
            qp.lonlat = (lon_clue, lat_clue)
            qp.lonlat_xp = (lon_clue, lat_clue)
            qp.lonlat_yp = (lon_clue, lat_clue)
            results.append(qp)
        return results

    def _clip(self, value: float, lower: float, upper: float) -> float:
        return lower if value < lower else upper if value > upper else value

    def _score_query_points(self, query_pts: list[QueryPoint]) -> float:
        print("scoring...")
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
            return None
        rmse = math.sqrt(sum_sq_error / num_pts)
        return rmse
