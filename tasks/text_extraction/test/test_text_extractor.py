from tasks.text_extraction.text_extractor import (
    ResizeTextExtractor,
    TileTextExtractor,
    DocTextExtraction,
)
from tasks.io import image_io
from pathlib import Path
import pytest


@pytest.mark.skip(reason="requires google vision credentials")
def test_resize_text_extractor():
    """
    Test function for ResizeTextExtractor class.

    Creates a ResizeTextExtractor object and tests its process() method
    on a test image. Asserts that the resulting doc_text_extraction object
    contains the expected text extractions and that the cache file is created
    and deleted successfully.
    """
    # create ResizeTextExtractor object
    cache_dir = Path("tasks/text_extraction/test/data")
    to_blocks = True
    document_ocr = False
    pixel_lim = 500
    rte = ResizeTextExtractor(cache_dir, to_blocks, document_ocr, pixel_lim)

    # test process()
    doc_id = "test"
    im = image_io.load_pil_image("tasks/text_extraction/test/data/test.jpg")
    doc_text_extraction = rte.process(doc_id, im)

    # check doc_text_extraction
    expected_doc_id = f"{doc_id}-google-cloud-visionresize-{pixel_lim}"
    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5

    validate_tile_extractions(doc_text_extraction)

    # check cache
    cache_file = cache_dir / f"{expected_doc_id}.json"
    assert cache_file.exists()
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
    cache_dir = Path("tasks/text_extraction/test/data")
    to_blocks = True
    document_ocr = False
    tile_size = 256
    tte = TileTextExtractor(cache_dir, tile_size)

    # test process()
    doc_id = "test"
    im = image_io.load_pil_image("tasks/text_extraction/test/data/test.jpg")
    doc_text_extraction = tte.process(doc_id, im)

    # check doc_text_extraction
    expected_doc_id = f"{doc_id}-google-cloud-visiontile-{tile_size}"
    assert doc_text_extraction.doc_id == expected_doc_id
    assert len(doc_text_extraction.extractions) == 5

    validate_tile_extractions(doc_text_extraction)

    # check cache
    cache_file = cache_dir / f"{expected_doc_id}.json"
    assert cache_file.exists()
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
