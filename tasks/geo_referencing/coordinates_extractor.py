import logging
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import Coordinate
from typing import Any, Dict, Tuple

logger = logging.getLogger("coordinates_extractor")


class CoordinateInput:
    input: TaskInput
    updated_output: Dict[Any, Any] = {}

    def __init__(self, input: TaskInput):
        self.input = input
        self.updated_output = {}


class CoordinatesExtractor(Task):
    """
    Generic task for extracting geo coordinates from a map
    """

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """

        input_coord = CoordinateInput(input)

        if not self._should_run(input_coord):
            return self._create_result(input_coord.input)

        # extract the coordinates using the input
        lats = input.get_data("lats", [])
        lons = input.get_data("lons", [])
        lons, lats = self._extract_coordinates(input_coord)
        logger.info(
            f"Num coordinates extracted: {len(lats)} latitude and {len(lons)} longitude"
        )

        # add the extracted coordinates to the result
        return self._create_coordinate_result(input_coord, lons, lats)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return {}, {}

    def _create_coordinate_result(
        self,
        input: CoordinateInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input.input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        for k, v in input.updated_output.items():
            result.output[k] = v

        return result

    def _should_run(self, input: CoordinateInput) -> bool:

        lats = input.input.get_data("lats", {})
        lons = input.input.get_data("lons", {})
        num_keypoints = min(len(lons), len(lats))

        # TODO: could check the number of lats and lons with status == OK
        # lats = list(filter(lambda c: c._status == CoordStatus.OK, lats.values()))
        # lons = list(filter(lambda c: c._status == CoordStatus.OK, lons.values()))

        return num_keypoints < 2
