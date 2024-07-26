import logging
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.point_extraction.entities import (
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
)
from tasks.point_extraction.legend_item_utils import (
    filter_labelme_annotations,
    LEGEND_ANNOTATION_PROVENANCE,
)
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY

logger = logging.getLogger(__name__)


class LegendPreprocessor(Task):
    """
    Pre-processing of Point Symbol Legend Items
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

        legend_pt_items = None
        if LEGEND_ITEMS_OUTPUT_KEY in task_input.data:
            # legend items for point symbols already exist
            legend_pt_items = LegendPointItems.model_validate(
                task_input.data[LEGEND_ITEMS_OUTPUT_KEY]
            )
        elif LEGEND_ITEMS_OUTPUT_KEY in task_input.request:
            # legend items for point symbols already exist as a request param
            # (ie, loaded from a JSON hints file)
            # convert to a TaskResult...
            legend_pt_items = LegendPointItems.model_validate(
                task_input.request[LEGEND_ITEMS_OUTPUT_KEY]
            )

        if (
            legend_pt_items
            and legend_pt_items.provenance == LEGEND_ANNOTATION_PROVENANCE.LABELME
        ):
            if SEGMENTATION_OUTPUT_KEY in task_input.data:
                segmentation = MapSegmentation.model_validate(
                    task_input.data[SEGMENTATION_OUTPUT_KEY]
                )
                # use segmentation results to filter noisy "labelme" legend annotations
                # (needed because all labelme annotations are set to type "polygon" regardless of feature type: polygons, lines or points)
                filter_labelme_annotations(legend_pt_items, segmentation)
                logger.info(
                    f"Number of legend point annotations after filtering: {len(legend_pt_items.items)}"
                )
            else:
                logger.warning(
                    "No segmentation results available. Disregarding labelme legend annotations as noisy."
                )
                legend_pt_items.items = []
            return TaskResult(
                task_id=self._task_id, output={LEGEND_ITEMS_OUTPUT_KEY: legend_pt_items}
            )

        return self._create_result(task_input)


class LegendPostprocessor(Task):
    """
    Post-processing of Point Symbol Legend Items
    """

    def __init__(
        self,
        task_id: str,
        cache_path: str,
    ):

        super().__init__(task_id, cache_path)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        run point symbol legend post-processing
        """
        ## WIP!

        return self._create_result(task_input)

