from tasks.metadata_extraction.text_filter import TextFilter
from tasks.common.task import TaskInput
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.segmentation.entities import SEGMENTATION_OUTPUT_KEY
from tasks.metadata_extraction.text_filter import FilterMode


def create_test_data() -> TaskInput:
    # create mock data
    text_data = {
        "doc_id": "test",
        "extractions": [
            {
                "text": "FUMIOUS",
                "confidence": 1.0,
                "bounds": [
                    {"x": 4, "y": 9},
                    {"x": 89, "y": 9},
                    {"x": 89, "y": 22},
                    {"x": 4, "y": 22},
                ],
            },
            {
                "text": "BANDERSNATCH SNICKER-SNACK",
                "confidence": 1.0,
                "bounds": [
                    {"x": 3, "y": 30},
                    {"x": 157, "y": 31},
                    {"x": 157, "y": 46},
                    {"x": 3, "y": 45},
                ],
            },
        ],
    }

    segments = {
        "doc_id": "test",
        "segments": [
            {
                "class_label": "legend_points_lines",
                "poly_bounds": [
                    [0, 0],
                    [100, 0],
                    [100, 100],
                    [0, 100],
                ],
                "area": 1.0,
                "bbox": [0, 0, 100, 100],
                "confidence": 1.0,
                "id_model": "test-model",
            },
            {
                "class_label": "map",
                "poly_bounds": [[101, 101], [200, 100], [200, 200], [101, 200]],
                "area": 1.0,
                "bbox": [101, 101, 200, 200],
                "confidence": 1.0,
                "id_model": "test-model",
            },
        ],
    }

    # create TaskInput with mock data
    input_data = {
        TEXT_EXTRACTION_OUTPUT_KEY: text_data,
        SEGMENTATION_OUTPUT_KEY: segments,
    }
    input = TaskInput(0)
    input.data = input_data

    return input


def test_text_filter_exclude():
    input = create_test_data()

    # create the TextFilter
    text_filter = TextFilter("test-text-filter", FilterMode.EXCLUDE)

    # run the TextFilter
    result = text_filter.run(input)

    # check the output
    assert len(result.output) == 1

    assert result.output[TEXT_EXTRACTION_OUTPUT_KEY] is not None

    doc_text_extraction = DocTextExtraction.model_validate(
        result.output[TEXT_EXTRACTION_OUTPUT_KEY]
    )
    assert len(doc_text_extraction.extractions) == 2

    extraction = doc_text_extraction.extractions[0]
    assert extraction.text == "FUMIOUS"

    extraction = doc_text_extraction.extractions[1]
    assert extraction.text == "BANDERSNATCH SNICKER-SNACK"


def test_text_filter_include():
    input = create_test_data()

    # create the TextFilter
    text_filter = TextFilter("test-text-filter", FilterMode.INCLUDE)

    # run the TextFilter
    result = text_filter.run(input)

    # check the output
    assert len(result.output) == 1

    assert result.output[TEXT_EXTRACTION_OUTPUT_KEY] is not None

    doc_text_extraction = DocTextExtraction.model_validate(
        result.output[TEXT_EXTRACTION_OUTPUT_KEY]
    )
    assert len(doc_text_extraction.extractions) == 2
    extraction = doc_text_extraction.extractions[0]
    assert extraction.text == "FUMIOUS"
    extraction = doc_text_extraction.extractions[1]
    assert extraction.text == "BANDERSNATCH SNICKER-SNACK"
