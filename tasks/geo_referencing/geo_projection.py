import logging
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tasks.geo_referencing.entities import Coordinate, CoordType

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

    def _map_coordinates(
        self, coords: list[Coordinate]
    ) -> Dict[Tuple[float, float], float]:
        # create the coordinate structure needed for point finalization
        coords_mapped = {}
        for c in coords:
            k, v = c.to_deg_result()
            coords_mapped[k] = v

        return coords_mapped
