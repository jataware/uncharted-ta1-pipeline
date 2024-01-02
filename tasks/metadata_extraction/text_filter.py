from pydoc import doc
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.text_extraction.entities import (
    TextExtraction,
    DocTextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.entities import MapSegmentation, SEGMENTATION_OUTPUT_KEY
from tasks.segmentation.detectron_segmenter import THING_CLASSES_DEFAULT
from enum import Enum
from shapely.geometry import Polygon
from typing import Dict


class FilterMode(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class TextFilter(Task):
    """
    Filter out text in map areas and legends
    """

    def __init__(self, task_id: str, filter_mode=FilterMode.EXCLUDE):
        self._filter_mode = filter_mode
        super().__init__("text_filter")

    def run(self, input: TaskInput) -> TaskResult:
        # get OCR output
        text_data = input.data[TEXT_EXTRACTION_OUTPUT_KEY]
        doc_text = DocTextExtraction.model_validate(text_data)

        # get map segments
        segments = input.data[SEGMENTATION_OUTPUT_KEY]
        map_segmentation = MapSegmentation.model_validate(segments)

        output_text: Dict[str, TextExtraction] = {}

        # filter out text in legends and map areas
        for text in doc_text.extractions:
            # create a shapely polygon from the text bounding box
            text_bounds_list = [(point.x, point.y) for point in text.bounds]
            text_poly = Polygon(text_bounds_list)
            hit = False
            # loop over map segments and check if the text intersects with any of them
            for segment in map_segmentation.segments:
                # create
                if segment.class_label in THING_CLASSES_DEFAULT:
                    segment_poly = Polygon(segment.poly_bounds)
                    if (
                        self._filter_mode == FilterMode.EXCLUDE
                        and segment_poly.contains(text_poly)
                    ):
                        hit = True
                        break
                    elif (
                        self._filter_mode == FilterMode.INCLUDE
                        and segment_poly.intersects(text_poly)
                    ):
                        hit = True
                        break
            # add the text to the output if it was not filtered out
            if (
                self._filter_mode == FilterMode.EXCLUDE
                and not hit
                or self._filter_mode == FilterMode.INCLUDE
                and hit
            ):
                output_text[text.text] = text

        doc_text.extractions = list(output_text.values())
        result = self._create_result(input)
        result.add_output(TEXT_EXTRACTION_OUTPUT_KEY, doc_text.model_dump())
        return result
