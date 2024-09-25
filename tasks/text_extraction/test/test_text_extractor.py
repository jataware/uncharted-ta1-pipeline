from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor
from tasks.text_extraction.entities import DocTextExtraction
from tasks.common import image_io
from pathlib import Path
from tasks.common.task import TaskInput
import pytest


@pytest.mark.skip(reason="requires google vision credentials")
def test_resize_text_extractor():
    """
    Test function for ResizeTextExtractor class.

    Creates a ResizeTextExtractor object and tests its process() method
    on a test image. Asserts that the resulting doc_text_extraction object
    contains the expected text extractions and that the cache file is created
    and re-used successfully.
    """
    # create ResizeTextExtractor object
    cache_dir = "tasks/text_extraction/test/data"
    to_blocks = True
    document_ocr = False
    pixel_lim = 500
    rte = ResizeTextExtractor(
        "resize_text", cache_dir, to_blocks, document_ocr, pixel_lim
    )

    # test process()
    doc_id = "test"
    im = image_io.load_pil_image("tasks/text_extraction/test/data/test.jpg")

    input = TaskInput(0)
    input.image = im
    input.raster_id = doc_id

    result = rte.run(input)
    doc_text_extraction = DocTextExtraction.model_validate(result.output)

    # check doc_text_extraction
    expected_doc_id = f"{doc_id}_google-cloud-vision_resize-{pixel_lim}"
    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5
    validate_tile_extractions(doc_text_extraction)

    # check cache
    cache_file = Path(cache_dir) / f"{expected_doc_id}.json"
    assert cache_file.exists()

    # re-run process() to test cached version
    result = rte.run(input)
    doc_text_extraction = DocTextExtraction.model_validate(result.output)
    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5
    validate_tile_extractions(doc_text_extraction)

    cache_file.unlink()


@pytest.mark.skip(reason="requires google vision credentials")
def test_tiling_text_extractor():
    """
    Test function for TilingTextExtractor class.

    Creates a TilingTextExtractor object and tests its process() method
    on a test image. Asserts that the resulting doc_text_extraction object
    contains the expected text extractions and that the cache file is created
    and deleted successfully.
    """
    # create TilingTextExtractor object
    cache_dir = "tasks/text_extraction/test/data"
    tile_size = 256
    tte = TileTextExtractor("tile_text", cache_dir, tile_size)

    # test process()
    doc_id = "test"
    im = image_io.load_pil_image("tasks/text_extraction/test/data/test.jpg")
    input = TaskInput(0)
    input.image = im
    input.raster_id = doc_id
    result = tte.run(input)
    doc_text_extraction = DocTextExtraction.model_validate(result.output)

    # check doc_text_extraction
    expected_doc_id = f"{doc_id}_google-cloud-vision_tile-{tile_size}"

    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5

    validate_tile_extractions(doc_text_extraction)

    # check cache
    cache_file = Path(cache_dir) / f"{expected_doc_id}.json"
    assert cache_file.exists()

    # re-run process() to test cached version
    result = tte.run(input)
    doc_text_extraction = DocTextExtraction.model_validate(result.output)
    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5
    validate_tile_extractions(doc_text_extraction)

    cache_file.unlink()


def validate_tile_extractions(doc_text_extraction: DocTextExtraction) -> None:
    # check first extraction
    extraction = doc_text_extraction.extractions[0]
    assert extraction.text == "FUMIOUS"
    assert extraction.confidence == 1.0
    assert len(extraction.bounds) == 4
    assert extraction.bounds[0].x == 4
    assert extraction.bounds[0].y == 9
    assert extraction.bounds[1].x == 89
    assert extraction.bounds[1].y == 9
    assert extraction.bounds[2].x == 89
    assert extraction.bounds[2].y == 22
    assert extraction.bounds[3].x == 4
    assert extraction.bounds[3].y == 22

    # check second extraction
    extraction = doc_text_extraction.extractions[1]
    assert extraction.text == "BANDERSNATCH"
    assert extraction.confidence == 1.0
    assert len(extraction.bounds) == 4
    assert extraction.bounds[0].x == 3
    assert extraction.bounds[0].y == 30
    assert extraction.bounds[1].x == 157
    assert extraction.bounds[1].y == 31
    assert extraction.bounds[2].x == 157
    assert extraction.bounds[2].y == 46
    assert extraction.bounds[3].x == 3
    assert extraction.bounds[3].y == 45
