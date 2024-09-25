import logging
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.point_extraction.entities import (
    LegendPointItems,
    PointLabels,
    LEGEND_ITEMS_OUTPUT_KEY,
    MAP_PT_LABELS_OUTPUT_KEY,
)

logger = logging.getLogger(__name__)


class FinalizePointExtractions(Task):
    """
    Final task for the point extraction pipeline.
    Joins point extractions and legend items into one pydantic object
    in preparation to mapping to the CDR's schema
    """

    def __init__(
        self,
        task_id: str,
    ):

        super().__init__(task_id)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Run the finalize point extractions task
        """

        # --- load legend item annotations, if available
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
        else:
            legend_pt_items = LegendPointItems(items=[])

        # --- get existing point predictions
        if MAP_PT_LABELS_OUTPUT_KEY in task_input.data:
            map_point_labels = PointLabels.model_validate(
                task_input.data[MAP_PT_LABELS_OUTPUT_KEY]
            )
            if map_point_labels.labels is None:
                map_point_labels.labels = []
        else:
            map_point_labels = PointLabels(
                path="", raster_id=task_input.raster_id, labels=[]
            )

        # join legend items with point extraction output (for easier conversion to CDR)
        map_point_labels.legend_items = legend_pt_items.items

        return TaskResult(
            task_id=self._task_id,
            output={MAP_PT_LABELS_OUTPUT_KEY: map_point_labels.model_dump()},
        )
