import logging
from tasks.geo_referencing.entities import Coordinate, CoordSource
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.geo_projection import PolyRegression
from typing import Dict, Tuple

logger = logging.getLogger("coordinates_filter")


class FilterCoordinates(Task):

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered, lat_pts_filtered = self._filter(input, lon_pts, lat_pts)

        logger.info(
            f"Num coordinates after filtering: {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)}"
        )

        # update the coordinates list
        return self._create_result(input, lon_pts_filtered, lat_pts_filtered)

    def _create_result(
        self,
        input: TaskInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return lon_coords, lat_coords


class FilterAxisCoordinates(Task):

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered = lon_pts
        if len(lon_pts) > 0:
            lon_pts_filtered = self._filter(input, lon_pts)
        lat_pts_filtered = lat_pts
        if len(lat_pts) > 0:
            lat_pts_filtered = self._filter(input, lat_pts)

        logger.info(
            f"Num coordinates after axis filtering: {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)}"
        )

        # update the coordinates list
        return self._create_result(input, lon_pts_filtered, lat_pts_filtered)

    def _create_result(
        self,
        input: TaskInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        return coords


class OutlierFilter(FilterAxisCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        if len(coords) < 3:
            logger.debug(
                "Skipping outlier filtering since there are fewer than 3 coordinates"
            )
            return coords

        logger.info(f"Running outlier filter for {len(coords)} coords")
        updated_coords = coords
        test_length = 0
        while len(updated_coords) != test_length:
            test_length = len(updated_coords)
            updated_coords = self._filter_regression(input, updated_coords)
            logger.debug(
                f"outlier filter updated length {len(updated_coords)} compared to test length {test_length}"
            )
        return updated_coords

    def _filter_regression(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # use leave one out approach using linear regression model

        # reduce coordinate to (degree, constant dimension) where the constant dimension for lat is y and lon is x
        coords_representation = []
        for _, c in coords.items():
            coords_representation.append(c)

        # get the regression quality when holding out each coordinate one at a time
        reduced = [
            self._reduce(coords_representation, i)
            for i in range(len(coords_representation))
        ]

        # identify potential outliers via the model quality (outliers should show a dip in the error)
        results = {}
        test = sum(reduced) / len(coords_representation)
        for i in range(len(coords_representation)):
            # arbitrary test to flag outliers
            # having a floor to the test prevents removing datapoints when the error is low due to points lining up correctly
            if test > 0.1 and reduced[i] < 0.5 * test:
                continue
            else:
                key, _ = coords_representation[i].to_deg_result()
                results[key] = coords_representation[i]
        return results

    def _reduce(self, coords: list[Coordinate], index: int) -> float:
        # remove the point for which to calculate the model quality
        coords_work = coords.copy()
        coords_work.pop(index)

        # build linear regression model using the remaining points
        regression = PolyRegression(1)
        pixels = []
        degrees = []
        for c in coords_work:
            pixels.append(c.get_pixel_alignment())
            degrees.append(c.get_parsed_degree())

        # do polynomial regression for axis
        regression.fit_polynomial_regression(pixels, degrees)
        predictions = regression.predict_pts(pixels)

        # calculate error
        # TODO: FOR NOW DO SIMPLE SUM
        return sum([abs(degrees[i] - predictions[i]) for i in range(len(predictions))])


class UTMStatePlaneFilter(FilterCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:

        # get the count and confidence of state plane and utm coordinates
        lon_count_sp, lon_conf_sp = self._get_score(lon_coords, CoordSource.STATE_PLANE)
        lon_count_utm, lon_conf_utm = self._get_score(lon_coords, CoordSource.UTM)
        lat_count_sp, lat_conf_sp = self._get_score(lat_coords, CoordSource.STATE_PLANE)
        lat_count_utm, lat_conf_utm = self._get_score(lat_coords, CoordSource.UTM)

        # if no utm or no state plane coordinates exist then nothing to filter
        if lon_count_sp + lat_count_sp == 0:
            return lon_coords, lat_coords
        if lon_count_utm + lat_count_utm == 0:
            return lon_coords, lat_coords

        # if one has coordinates in both directions while the other doesnt then keep that one
        source_filter = None
        if (
            min(lon_count_utm, lat_count_utm) > 0
            and min(lon_count_sp, lat_count_sp) == 0
        ):
            logger.debug("removing state plane coordinates since one axis has none")
            source_filter = CoordSource.STATE_PLANE
        elif (
            min(lon_count_utm, lat_count_utm) == 0
            and min(lon_count_sp, lat_count_sp) > 0
        ):
            logger.debug("removing utm coordinates since one axis has none")
            source_filter = CoordSource.UTM

        # if still unsure then retain the one with the highest confidence
        # by this point both utm and state plane have coordinates in one or two directions
        if not source_filter:
            source_filter = CoordSource.UTM
            if max(lon_conf_utm, lat_conf_utm) > max(lon_conf_sp, lat_conf_sp):
                logger.debug(
                    "removing state plane coordinates since utm coordinates have higher confidence"
                )
                source_filter = CoordSource.STATE_PLANE
            else:
                logger.debug(
                    "removing utm coordinates since state plane coordinates have higher confidence"
                )

        logger.debug(f"filtering {source_filter} latitude and longitude coordinates")

        return self._filter_source(source_filter, lon_coords), self._filter_source(
            source_filter, lat_coords
        )

    def _get_score(
        self, coords: Dict[Tuple[float, float], Coordinate], source: CoordSource
    ) -> Tuple[int, float]:
        conf = -1
        count = 0
        for _, c in coords.items():
            src = c.get_source()
            if src == source:
                count = count + 1
                if conf < c.get_confidence():
                    conf = c.get_confidence()
        return (count, conf)

    def _filter_source(
        self, source: CoordSource, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        coords_filtered = {}
        for k, c in coords.items():
            if not c.get_source() == source:
                coords_filtered[k] = c
        return coords_filtered
