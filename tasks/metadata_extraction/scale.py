import csv
import logging

from pydoc import doc
from tasks.common.task import Task, TaskInput, TaskResult
from typing import List, Tuple

SCALE_VALUE_OUTPUT_KEY = "scale_value"

logger = logging.getLogger("scale_extractor")


class ScaleExtractor(Task):
    """
    Extract the scale from the image initially by reading from a csv file
    """

    def __init__(self, task_id: str, file_name: str):
        self._file_name = file_name
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        logger.info(
            f"running scale extractor with id {self._task_id} on image {input.raster_id}"
        )

        # read the scale file
        map_scales = self._read_map_scales(self._file_name)

        # get the scale string
        scale = self._extract_scale(input.raster_id, map_scales)

        # parse the scale string to get the float value
        scale_value = -1
        if len(scale) > 2:
            scale_value = self._parse_scale(scale)

        # generate the result
        result = self._create_result(input)
        result.output[SCALE_VALUE_OUTPUT_KEY] = scale_value
        logger.info(
            f"scale extractor ({self._task_id}) extracted scale {scale_value} for image {input.raster_id}"
        )

        return result

    def _parse_scale(self, scale: str) -> float:
        # assume scale format is "1:VALUE" and extract VALUE into a float
        return float(scale[2:])

    def _extract_scale(self, raster_id: str, map_scales: List[Tuple[str, str]]) -> str:
        # extract the scale for the specified map
        # assume the scales are (RASTER ID, SCALE)
        for s in map_scales:
            if s[0] == raster_id:
                return s[1]
        # not found so default to enpty string
        return ""

    def _read_map_scales(self, file_name: str) -> List[Tuple[str, str]]:
        # read the csv file assuming RASTER_ID,SCALE
        map_scales: List[Tuple[str, str]] = []

        file = open(file_name, "r")
        data_raw: List[List[str]] = list(csv.reader(file, delimiter=","))
        file.close()

        # skip header, put into expected list format
        for dr in data_raw[1:]:
            map_scales.append((dr[0], dr[1]))

        return map_scales
