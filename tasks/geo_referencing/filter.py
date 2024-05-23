import logging
import uuid

import numpy as np

from sklearn.cluster import DBSCAN

from tasks.geo_referencing.entities import Coordinate, SOURCE_STATE_PLANE, SOURCE_UTM
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.geo_projection import PolyRegression
from tasks.geo_referencing.util import ocr_to_coordinates

from typing import Dict, Tuple

logger = logging.getLogger("coordinates_filter")

NAIVE_FILTER_MINIMUM = 10


class FilterCoordinates(Task):
    _coco_file_path: str = ""
    _buffering_func = None

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")
        logger.info(
            f"prior to filtering {len(lat_pts)} latitude and {len(lon_pts)} longitude coordinates have been extracted"
        )

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered = self._filter(input, lon_pts)
        lat_pts_filtered = self._filter(input, lat_pts)
        logger.info(
            f"after filtering run {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)} longitude coordinates have been retained"
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


class OutlierFilter(FilterCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        logger.info(f"outlier filter running against {coords}")
        updated_coords = coords
        test_length = 0
        while len(updated_coords) != test_length:
            test_length = len(updated_coords)
            updated_coords = self._filter_regression(input, updated_coords)
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
                self._add_param(
                    input,
                    str(uuid.uuid4()),
                    "coordinate-excluded",
                    {
                        "bounds": ocr_to_coordinates(
                            coords_representation[i].get_bounds()
                        ),
                        "text": coords_representation[i].get_text(),
                        "type": (
                            "latitude"
                            if coords_representation[i].is_lat()
                            else "longitude"
                        ),
                        "pixel_alignment": coords_representation[
                            i
                        ].get_pixel_alignment(),
                        "confidence": coords_representation[i].get_confidence(),
                    },
                    "excluded due to regression outlier detection",
                )
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
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        logger.info(f"utm - state plane filter running against {coords}")

        # figure out which of utm and state plane have the higher confidence
        conf_sp = -1
        conf_utm = -1
        count_sp = 0
        for _, c in coords.items():
            src = c.get_source()
            if src == SOURCE_STATE_PLANE:
                count_sp = count_sp + 1
                if conf_sp < c.get_confidence():
                    conf_sp = c.get_confidence()
            if src == SOURCE_UTM:
                count_sp = count_sp - 1
                if conf_utm < c.get_confidence():
                    conf_utm = c.get_confidence()

        # retain the one with higher confidence
        source_filter = ""
        if conf_utm >= 0 and conf_sp >= 0:
            source_filter = SOURCE_UTM
            if conf_utm > conf_sp:
                source_filter = SOURCE_STATE_PLANE
            elif conf_utm == conf_sp:
                # use the count to determine which to filter
                if count_sp < 0:
                    source_filter = SOURCE_STATE_PLANE
            logger.info(f"removing coordinates with source {source_filter}")

        return self._filter_source(source_filter, coords)

    def _filter_source(
        self, source: str, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        coords_filtered = {}
        for k, c in coords.items():
            if not c.get_source() == source:
                coords_filtered[k] = c
        return coords_filtered


class NaiveFilter(FilterCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        logger.info(f"naive filter running against {coords}")
        updated_coords = self._filter_coarse(input, coords)
        return updated_coords

    def _filter_coarse(
        self, input: TaskInput, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        # check range of coordinates to determine if filtering is required
        degs = []
        for _, c in coords.items():
            degs.append(c.get_parsed_degree())

        if max(degs) - min(degs) < NAIVE_FILTER_MINIMUM:
            return coords

        # cluster degrees
        data = np.array([[d] for d in degs])
        db = DBSCAN(eps=2.5, min_samples=2).fit(data)
        labels = db.labels_

        clusters = []
        max_cluster = []
        for i, l in enumerate(labels):
            if l == -1:
                continue
            while len(clusters) <= l:
                clusters.append([])
            clusters[l].append(degs[i])
            if len(clusters[l]) > len(max_cluster):
                max_cluster = clusters[l]

        # no clustering so unable to filter anything reliably
        if len(max_cluster) == 0:
            return coords

        filtered_coords = {}
        for k, v in coords.items():
            if v.get_parsed_degree() in max_cluster:
                filtered_coords[k] = v
            else:
                self._add_param(
                    input,
                    str(uuid.uuid4()),
                    "coordinate-excluded",
                    {
                        "bounds": ocr_to_coordinates(v.get_bounds()),
                        "text": v.get_text(),
                        "type": ("latitude" if v.is_lat() else "longitude"),
                        "pixel_alignment": v.get_pixel_alignment(),
                        "confidence": v.get_confidence(),
                    },
                    "excluded due to naive outlier detection",
                )
        return filtered_coords
