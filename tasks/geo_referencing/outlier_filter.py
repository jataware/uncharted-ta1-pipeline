import logging, math
import numpy as np
from shapely import Polygon
from sklearn.linear_model import RANSACRegressor, LinearRegression
from tasks.geo_referencing.entities import (
    DocGeoFence,
    GeoFenceType,
    MapScale,
    GEOFENCE_OUTPUT_KEY,
    MAP_SCALE_OUTPUT_KEY,
    ROI_MAP_OUTPUT_KEY,
    MapROI,
    Coordinate,
    CoordStatus,
    CoordSource,
)

from tasks.geo_referencing.util import (
    sign,
    is_coord_from_source,
    calc_lonlat_slope_signs,
)
from tasks.common.task import Task, TaskInput, TaskResult
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("outlier_filter")

# max skew (in degrees) of a map to account for during outlier detection
MAX_SKEW_DEFAULT = 20
# multiplier to check the expected degrees-to-pixel scale
DEG2PIXEL_VALIDITY_FACTOR = 5


class OutlierFilter(Task):
    """
    Outlier analysis of the extracted coordinates using regression methods
    """

    def __init__(
        self,
        task_id: str,
        coord_source_check: List[CoordSource] = [],
        max_skew_angle: float = MAX_SKEW_DEFAULT,
        force_deg_abs: bool = True,
    ):
        self._coord_source_check = coord_source_check
        self._max_skew_angle = max_skew_angle
        self._force_deg_abs = force_deg_abs
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """
        # get geofence result
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        # get map-scale result
        map_scale: MapScale = input.parse_data(
            MAP_SCALE_OUTPUT_KEY, MapScale.model_validate
        )
        # get map-roi result
        map_roi: MapROI = input.parse_data(ROI_MAP_OUTPUT_KEY, MapROI.model_validate)

        # get coordinates so far
        lons = input.get_data("lons", {})
        lats = input.get_data("lats", {})
        lons_excluded = input.get_data("lons_excluded", {})
        lats_excluded = input.get_data("lats_excluded", {})
        if self._coord_source_check:
            if not is_coord_from_source(
                lons, self._coord_source_check
            ) and not is_coord_from_source(lats, self._coord_source_check):
                # "No coordinates are from extractors {self._coord_source_check}; skipping outlier filtering
                return self._create_result(input)

        # get expected sign of the regression slope(s) (pixels to degrees)
        lonlat_slopes_expected = map_scale.lonlat_per_pixel if map_scale else (0.0, 0.0)
        lonlat_slope_signs = (0, 0)
        if geofence and not geofence.geofence.region_type == GeoFenceType.DEFAULT:
            lonlat_slope_signs = calc_lonlat_slope_signs(
                geofence.geofence.lonlat_hemispheres
            )
        # estimate the lon and lat max skew thresholds
        lonlat_skew_thresholds = self._calc_skew_thresholds(map_roi)

        # outlier analysis for longitudes
        lons, lons_excluded, lons_slope = self._regression_analysis(
            lons,
            lons_excluded,
            lonlat_slopes_expected[0],
            lonlat_slope_signs[1],
            lonlat_skew_thresholds[1],
        )

        # outlier analysis for latitudes
        lats, lats_excluded, lats_slope = self._regression_analysis(
            lats,
            lats_excluded,
            lonlat_slopes_expected[1],
            lonlat_slope_signs[0],
            lonlat_skew_thresholds[0],
        )

        # update the coordinates lists
        result = self._create_result(input)
        result.output["lons"] = lons
        result.output["lats"] = lats
        result.output["lons_excluded"] = lons_excluded
        result.output["lats_excluded"] = lats_excluded

        return result

    def _regression_analysis(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        coords_excluded: Dict[Tuple[float, float], Coordinate],
        slope_expected: float = 0.0,
        slope_sign: int = 0,
        skew_thres: float = 0.0,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate],
        Dict[Tuple[float, float], Coordinate],
        Optional[float],
    ]:
        """
        Regression analysis of the extracted coords
        Output is a tuple of the form:
            coords,
            coords_exluded (ie outliers),
            regression slope
        """

        def is_data_valid(X: np.ndarray, y: np.ndarray) -> bool:
            """
            Child function -- Data subset validation function to use with regression analysis
            I.e., used to restrict the slope best-fit close to the target expected value
            """
            if slope_sign == 0:
                return True
            try:
                # do linear regression of data subset and check the slope
                linreg = LinearRegression().fit(X, y)
                lin_slope = linreg.coef_[0]
                ok = False
                if slope_expected != 0.0:
                    ok = (
                        sign(slope_sign) == sign(lin_slope)
                        and slope_expected / DEG2PIXEL_VALIDITY_FACTOR < abs(lin_slope)
                        and slope_expected * DEG2PIXEL_VALIDITY_FACTOR > abs(lin_slope)
                    )
                    return ok
                else:
                    ok = sign(slope_sign) == sign(lin_slope)
                return ok
            except Exception as ex:
                logger.warning(
                    f"Exception with is_data_valid func; returning False -- {repr(ex)}"
                )
                return False

        min_coords = 4 if slope_expected == 0.0 else 3
        if len(coords) < min_coords:
            # skip outlier analysis, not enough points
            return (coords, coords_excluded, None)

        try:
            # --- do regression analysis 2D pixel values (x) vs coord degrees (y)
            X = []
            y = []
            for c in coords.values():
                y.append(c.get_parsed_degree())
                pixel_xy = c.get_pixel_alignment()
                if c.is_lat():
                    X.append(pixel_xy[1])
                else:
                    X.append(pixel_xy[0])

            # convert to numpy arrays (note, X must be 2D)
            X = np.array(X).reshape((-1, 1))
            y = np.array(y)
            if self._force_deg_abs:
                y = np.abs(y)

            residual_threshold = self._calc_residual_threshold(y)

            data_checker_func = is_data_valid if slope_sign != 0 else None

            regressor = RANSACRegressor(
                random_state=911,
                residual_threshold=residual_threshold,
                is_data_valid=data_checker_func,
            )
            try:
                regressor.fit(X, y)
            except Exception as ex_reg:
                logger.debug(
                    "Exception during regression calc; re-try without slope value contraint"
                )
                slope_expected = 0.0
                regressor.fit(X, y)

            inlier_mask = self._check_x_residuals(regressor, X, y, skew_thres)
            outlier_mask = np.logical_not(inlier_mask)

            # remove any outlier coords
            num_outliers = len(y[outlier_mask])
            if num_outliers > 0:
                # outliers found; remove them
                logger.info(f"Num outliers found: {num_outliers}")
                c_keys = list(coords.keys())
                for idx, is_out in enumerate(outlier_mask.tolist()):
                    if is_out:
                        # remove outlier coord
                        excl_key = c_keys[idx]
                        excl_c = coords.pop(excl_key)
                        excl_c._status = CoordStatus.OUTLIER
                        coords_excluded[excl_key] = excl_c

            # get the best-fit slope of the regression
            slope = regressor.estimator_.coef_[0]  # type: ignore

        except Exception as ex:
            logger.warning(
                f"Exception with outlier regression analysis; keeping all coords -- {repr(ex)}"
            )
            return (coords, coords_excluded, None)

        return (coords, coords_excluded, slope)

    def _calc_skew_thresholds(self, map_roi: Optional[MapROI]) -> Tuple[float, float]:
        """
        estimate the max pixel corner offset (in x and y directions), based on map size and a max skew angle
        """
        if not map_roi:
            return (0, 0)
        roi_poly = Polygon(map_roi.map_bounds)
        w = roi_poly.bounds[2] - roi_poly.bounds[0]
        h = roi_poly.bounds[3] - roi_poly.bounds[1]
        # (1/2 factor below due to +ve and -ve residuals for regression result on a skewed map)
        lon_thres = w * math.sin(self._max_skew_angle * math.pi / 180) / 2
        lat_thres = h * math.sin(self._max_skew_angle * math.pi / 180) / 2
        return (lon_thres, lat_thres)

    def _calc_residual_threshold(self, y: np.ndarray) -> float:
        """
        Calc the RANSAC residual threshold
        Based on median absolute deviation (MAD) of y values
        """
        mad_factor = 1.0
        if len(y) > 4:
            # MAD factor (for median absolute deviation) -- use a more strict outlier threshold if 4 > pts
            mad_factor = max(4.0 / len(y), 0.5)
        # MAD calc
        residual_threshold = np.median(np.abs(y - np.median(y)))
        if residual_threshold == 0.0:
            # residual threshold too strict! Use average calc instead
            residual_threshold = np.average(np.abs(y - np.median(y)))
        residual_threshold *= mad_factor
        return float(residual_threshold)

    def _check_x_residuals(
        self,
        regressor: RANSACRegressor,
        X: np.ndarray,
        y: np.ndarray,
        x_residual_thres: float,
    ) -> np.ndarray:
        """
        check residuals in the x-direction using skew angle threshold
        """
        if x_residual_thres == 0.0 or regressor.inlier_mask_.all():
            # no need to check
            return regressor.inlier_mask_

        slope = regressor.estimator_.coef_[0]  # type: ignore
        intercept = regressor.estimator_.intercept_  # type: ignore
        if not slope:
            return regressor.inlier_mask_

        inlier_flags = []
        for idx, inlier in enumerate(regressor.inlier_mask_):
            if not inlier:
                # double-check x-residual of this outlier
                x_pred = (y[idx] - intercept) / slope
                x_error = abs(X[idx][0] - x_pred)
                if x_error >= x_residual_thres:
                    inlier_flags.append(False)
                    continue
            inlier_flags.append(True)

        return np.array(inlier_flags)
