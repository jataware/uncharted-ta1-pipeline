import logging
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.point_extraction.entities import (
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
)

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

        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            # legend items for point symbols already exist
            result = self._create_result(task_input)
            return result
        elif LEGEND_ITEMS_OUTPUT_KEY in task_input.request:
            # legend items for point symbols already exist as a request param
            # (ie, loaded from a JSON hints file)
            # convert to a TaskResult...
            legend_pt_items = LegendPointItems.model_validate(
                task_input.request[LEGEND_ITEMS_OUTPUT_KEY]
            )
            return TaskResult(
                task_id=self._task_id, output={LEGEND_ITEMS_OUTPUT_KEY: legend_pt_items}
            )

        # TODO WIP
        # Continue here...
        # If no legend item hints available then could do our own naive version using segmentation, ocr, etc.?

        result = self._create_result(task_input)
        return result
