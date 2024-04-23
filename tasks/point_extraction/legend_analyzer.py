import logging

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.point_extraction.entities import (
    LegendPointItem,
    LEGEND_ITEMS_OUTPUT_KEY,
)

# class labels for map and points legend areas
# SEGMENT_MAP_CLASS = "map"
SEGMENT_PT_LEGEND_CLASS = "legend_points_lines"

logger = logging.getLogger(__name__)


class PointLegendAnalyzer(Task):
    """
    Analysis of Point Symbol Legend Items
    """

    def __init__(
        self,
        task_id: str,
        cache_path: str,
    ):

        super().__init__(task_id, cache_path)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run point symbol legend analysis
        """

        if (
            LEGEND_ITEMS_OUTPUT_KEY in task_input.data
            or LEGEND_ITEMS_OUTPUT_KEY in task_input.request
        ):
            # legend items for point symbols already exist
            result = self._create_result(task_input)
            return result

        # TODO TEMP WIP
        # Continue here...
        # If no legend item hints available then do our own naive version using segmentation, ocr, etc.
        # legend_items = List of LegendPointItems(items=legend_point_items, provenance="modelled")
        # return TaskResult(
        #    task_id=self._task_id, output={LEGEND_ITEMS_OUTPUT_KEY: legend_items  ) ##result_map_tiles.model_dump()}
        # )

        # TODO TEMP
        result = self._create_result(task_input)
        return result
