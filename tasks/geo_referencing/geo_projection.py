import logging
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tasks.geo_referencing.entities import Coordinate

from typing import Dict, List, Tuple

logger = logging.getLogger("geo_projection")

X_MAX: float = 180
X_MIN: float = -180
Y_MAX: float = 90
Y_MIN: float = -90


class PolyRegression:
    def __init__(self, order: int):
        # note: better to set include_bias=False since this is handled for in LinearRegression model
        # see https://stackoverflow.com/questions/59725907/scikit-learn-polynomialfeatures-what-is-the-use-of-the-include-bias-option
        # for explanation
        self.polyreg = PolynomialFeatures(degree=order, include_bias=False)
        self.polyreg_model = LinearRegression()

    #
    # fit_polynomial_regression
    # (from this example: https://data36.com/polynomial-regression-python-scikit-learn/)
    #
    def fit_polynomial_regression(
        self, inputs_pts: List[List[float]], target_outputs: List[float]
    ):
        # self.polyreg = PolynomialFeatures(degree=order, include_bias=False)
        # self.polyreg_model = LinearRegression()
        poly_features = self.polyreg.fit_transform(np.array(inputs_pts))
        self.polyreg_model.fit(poly_features, target_outputs)

        # TODO try/catch here?

    def predict_pts(self, inputs_pts: List[Tuple[float, float]]) -> List[float]:
        predicted_outputs = self.polyreg_model.predict(
            self.polyreg.fit_transform(np.array(inputs_pts))
        )
        return predicted_outputs.tolist()


class GeoProjection:
    def __init__(self, order: int = 2):
        self.regression_X = PolyRegression(order)
        self.regression_Y = PolyRegression(order)

    #
    # estimate_pxl2geo_mapping
    #
    def estimate_pxl2geo_mapping(
        self,
        lon_coords: List[Coordinate],
        lat_coords: List[Coordinate],
        im_size: Tuple[int, int],
    ):
        # Use polynomial regression to
        # estimate x-pxl -> longitude and y-pxl -> latitude mapping, independently
        # BUT each mapping may depend on both x,y values for a given lon or lat value, respectively
        # (due to possible map rotation, or geo-projection warping, etc.)
        #
        # (Note: experiments trying joint 2D polynomialtransform produced erratic results for
        # a sparse number of keypoints)
        lon_results = self._map_coordinates(lon_coords)
        lat_results = self._map_coordinates(lat_coords)
        lon_results = self.finalize_keypoints(lon_results, im_size[0], im_size[1])
        lat_results = self.finalize_keypoints(lat_results, im_size[1], im_size[0])

        lon_xy = []
        lon_pts = []
        lat_xy = []
        lat_pts = []
        # x -> longitude
        for (lon, x), y in lon_results.items():
            lon_xy.append([x, y])
            lon_pts.append(lon)
        # y -> latitude
        for (lat, y), x in lat_results.items():
            lat_xy.append([x, y])
            lat_pts.append(lat)

        # do polynomial regression for x->longitude
        self.regression_X.fit_polynomial_regression(lon_xy, lon_pts)

        # do polynomial regression for y->latitude
        self.regression_Y.fit_polynomial_regression(lat_xy, lat_pts)

    #
    # predict_xy_pts
    #
    def predict_xy_pts(
        self, xy_pts: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        x_warped = self.regression_X.predict_pts(xy_pts)
        y_warped = self.regression_Y.predict_pts(xy_pts)

        # limit to geo coordinate range
        if min(x_warped) < X_MIN or max(x_warped) > X_MAX:
            logger.info(
                "adjusting longitude predictions due to values exceeding geographic range"
            )
            x_warped = [min(X_MAX, max(X_MIN, x)) for x in x_warped]
        if min(y_warped) < Y_MIN or max(y_warped) > Y_MAX:
            logger.info(
                "adjusting latitude predictions due to values exceeding geographic range"
            )
            y_warped = [min(Y_MAX, max(Y_MIN, x)) for x in y_warped]

        xy_warped = [(x_w, y_w) for x_w, y_w in zip(x_warped, y_warped)]

        return xy_warped

    #
    # finalize_keypoints
    #
    def finalize_keypoints(
        self, deg_results: Dict[Tuple[float, float], float], i_max: int, j_max: int
    ) -> Dict[Tuple[float, float], float]:
        # num of unique degree values
        num_deg_vals = len(set([x[0] for x in deg_results]))

        if num_deg_vals < 2:
            return deg_results
        if num_deg_vals == 2:
            # lon = (deg, x) y
            # lat = (deg, y) x
            # only 2 unique keypoints; not enough to reliably handle rotation in geo-projection,
            # so assume no rotation/skew of map, and add a 3rd keypoint to help 'anchor'
            # the polynomial regression fit (prevent erratic mapping behaviour)
            (deg, pxl_i), pxl_j = list(deg_results.items())[0]  # get 1st keypoint
            new_j = (
                0 if pxl_j > j_max / 2 else j_max - 1
            )  # new j value (far from the others)
            new_i = pxl_i + 1  # jitter by 1 pixel
            deg_results[(deg, new_i)] = new_j
            logger.info(
                "Adding an anchor keypoint (assume no skew): deg: {}, i,j: {},{}".format(
                    deg, new_i, new_j
                )
            )
            return deg_results

        # more than 2 unique keypoints, check if they are are all approx aligned in i-axis
        j_vals = [x[1] for x in deg_results.items()]
        j_delta = j_max * 0.05
        i_delta = i_max * 0.05
        if max(j_vals) - min(j_vals) < j_delta:
            # approx aligned along i-axis (< 5% j delta)
            # so assume "minimal" rotation/skew of map, and add an extra keypoint to help 'anchor'
            # the polynomial regression fit (prevent erratic mapping behaviour)
            i_vals = [x[0][1] for x in deg_results.items()]
            m, b = np.polyfit(
                i_vals, j_vals, 1
            )  # esimate rotation (slope) of map wrt i-axis

            (deg, pxl_i), pxl_j = list(deg_results.items())[0]  # get 1st keypoint

            new_j = (
                0 if pxl_j > j_max / 2 else j_max - 1
            )  # new j value (far from the others)
            i_offset = int(m * (pxl_j - new_j))
            if i_offset == 0:  # or i_offset > i_delta or i_offset < -i_delta:
                i_offset = 1  # jitter by 1 pixel (at least)
            new_i = max(min(pxl_i + i_offset, i_max - 1), 0)
            deg_results[(deg, new_i)] = new_j
            logger.info(
                "Adding an anchor keypoint (minimal skew): deg: {}, i,j: {},{}".format(
                    deg, new_i, new_j
                )
            )

        return deg_results

    def _map_coordinates(
        self, coords: list[Coordinate]
    ) -> Dict[Tuple[float, float], float]:
        # create the coordinate structure needed for point finalization
        coords_mapped = {}
        for c in coords:
            k, v = c.to_deg_result()
            coords_mapped[k] = v

        return coords_mapped
