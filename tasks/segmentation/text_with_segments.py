import logging
from shapely import Polygon
from collections import defaultdict


from tasks.text_extraction.entities import (
    DocTextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.entities import (
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
    SEGMENT_POLYGON_LEGEND_CLASS,
    SEGMENT_POINT_LEGEND_CLASS,
)
from tasks.common.task import Task, TaskInput, TaskResult


logger = logging.getLogger(__name__)


class TextWithSegments(Task):
    """
    Task to append OCR text extractions to Segmentation results, if available
    Text is added in an un-structured block
    """

    def __init__(self, task_id: str):

        self._segment_classes = [
            SEGMENT_POLYGON_LEGEND_CLASS,
            SEGMENT_POINT_LEGEND_CLASS,
        ]

        super().__init__(task_id)

    def run(self, task_input: TaskInput) -> TaskResult:
        """
        Run the text-with-segments task
        """

        # get OCR output
        doc_text = (
            DocTextExtraction.model_validate(
                task_input.data[TEXT_EXTRACTION_OUTPUT_KEY]
            )
            if TEXT_EXTRACTION_OUTPUT_KEY in task_input.data
            else DocTextExtraction(doc_id=task_input.raster_id, extractions=[])
        )

        if len(doc_text.extractions) == 0:
            logger.warning(
                f"No OCR data available for raster {task_input.raster_id}; skipping the TextWithSegments task."
            )
            result = self._create_result(task_input)
            return result

        # get the segmentation output
        if SEGMENTATION_OUTPUT_KEY not in task_input.data:
            logger.warning(
                f"No segmentation available for raster {task_input.raster_id}; skipping the TextWithSegments task."
            )
            result = self._create_result(task_input)
            return result

        segmentation = MapSegmentation.model_validate(
            task_input.data[SEGMENTATION_OUTPUT_KEY]
        )

        text_segments = defaultdict(str)

        for text in doc_text.extractions:
            text_bounds_list = [(point.x, point.y) for point in text.bounds]
            text_poly = Polygon(text_bounds_list)

            for i, seg in enumerate(segmentation.segments):
                if seg.class_label in self._segment_classes:
                    # try to append OCR text to this segment result...
                    segment_poly = Polygon(seg.poly_bounds)

                    if segment_poly.intersects(text_poly):
                        # this text block intersects with this segment,
                        # append text block
                        text_segments[i] += text.text + " "
                        break

        # append final merged OCR blocks with the segments
        for i, seg in enumerate(segmentation.segments):
            text_blk = text_segments.get(i, "")
            if text_blk:
                seg.text = text_blk

        result = self._create_result(task_input)
        result.add_output(SEGMENTATION_OUTPUT_KEY, segmentation.model_dump())
        return result
