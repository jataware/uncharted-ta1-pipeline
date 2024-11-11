import logging

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
from typing import Callable, Dict, List, Optional, Set


class FilterMode(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


logger = logging.getLogger("text_filter")


class TextFilter(Task):
    """
    Filter out text in map areas and legends
    """

    def __init__(
        self,
        task_id: str,
        filter_mode=FilterMode.EXCLUDE,
        input_key: str = TEXT_EXTRACTION_OUTPUT_KEY,
        output_key: str = TEXT_EXTRACTION_OUTPUT_KEY,
        classes: List[str] = THING_CLASSES_DEFAULT,
        class_threshold: int = 0,
        should_run: Optional[Callable] = None,
    ):
        self._filter_mode = filter_mode
        self._input_key = input_key
        self._output_key = output_key
        self._filering_classes = classes
        self._should_run = should_run
        self._class_threshold = class_threshold
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        if self._should_run and not self._should_run(input):
            return self._create_result(input)

        # get OCR output
        text_data = input.data[self._input_key]
        doc_text = DocTextExtraction.model_validate(text_data)

        # get map segments
        segments = input.data.get(
            SEGMENTATION_OUTPUT_KEY, {"doc_id": input.raster_id, "segments": []}
        )
        map_segmentation = MapSegmentation.model_validate(segments)

        output_text: Dict[str, TextExtraction] = {}

        # track the number of words contained by each segment - if there are less than
        # the class threshold, the segment will be excluded from filtering and all contained
        # text will be included in the output
        segment_words: Dict[str, int] = {}

        hits: Dict[str, List[bool]] = {
            seg.class_label: [False] * len(doc_text.extractions)
            for seg in map_segmentation.segments
        }

        # filter out text in legends and map areas
        for idx, text in enumerate(doc_text.extractions):
            # create a shapely polygon from the text bounding box
            text_bounds_list = [(point.x, point.y) for point in text.bounds]
            text_poly = Polygon(text_bounds_list)
            hit = False
            # loop over map segments and check if the text intersects with any of them
            for segment in map_segmentation.segments:
                # create
                if segment.class_label in self._filering_classes:
                    segment_poly = Polygon(segment.poly_bounds)

                    # first check - if the text is completely inside the segment add it to the
                    # segments word count
                    if segment_poly.contains(text_poly):
                        segment_words[segment.class_label] = (
                            segment_words.get(segment.class_label, 0) + 1
                        )

                    if (
                        self._filter_mode == FilterMode.EXCLUDE
                        and segment_poly.contains(text_poly)
                    ):
                        hits[segment.class_label][idx] = True
                    elif (
                        self._filter_mode == FilterMode.INCLUDE
                        and segment_poly.intersects(text_poly)
                    ):
                        hits[segment.class_label][idx] = True

        # disable filtering for segments with less than the class threshold word count
        skip_segments: Set[str] = set()
        for segment_class, word_count in segment_words.items():
            if word_count < self._class_threshold:
                skip_segments.add(segment_class)
                logger.info(
                    f"Skipping filtering for segment class {segment_class} with {word_count} words"
                )

        # flag the text as being included in the output if it was not filtered out
        for idx, text in enumerate(doc_text.extractions):
            for segment_class in hits.keys():
                if (
                    self._filter_mode == FilterMode.EXCLUDE
                    # skip filering segments with less than the class threshold word count in the exclude mo
                    and (not hits[segment_class][idx] or segment_class in skip_segments)
                ) or (
                    self._filter_mode == FilterMode.INCLUDE and hits[segment_class][idx]
                ):
                    output_text[text.text] = text

        doc_text.extractions = list(output_text.values())
        result = self._create_result(input)
        result.add_output(self._output_key, doc_text.model_dump())
        return result
