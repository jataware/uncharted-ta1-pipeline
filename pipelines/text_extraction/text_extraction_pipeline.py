import os
from pathlib import Path
import tqdm
from typing import Iterator, Tuple, List
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor
from tasks.text_extraction.entities import DocTextExtraction
from PIL.Image import Image as PILImage


class TextExtractionPipeline:
    """
    Pipeline for extracting text from images using OCR.

    Args:
        work_dir (Path): The directory where OCR output will be saved.
        tile (bool): Whether to tile the image before OCR.
        verbose (bool): Whether to print verbose output.

    Returns:
        List[DocTextExtraction]: A list of DocTextExtraction objects containing the extracted text.
    """

    def __init__(
        self,
        work_dir: Path,
        tile=False,
        verbose=False,
    ):
        self._ocr_output = Path(os.path.join(work_dir, "ocr_output"))
        self._verbose = verbose
        self._tile = tile

    def run(self, input: Iterator[Tuple[str, PILImage]]) -> List[DocTextExtraction]:
        """Runs OCR on the supplied stream of input images"""
        results: List[DocTextExtraction] = []

        if self._tile:
            ocr_task = TileTextExtractor(
                self._ocr_output,
            )
        else:
            ocr_task = ResizeTextExtractor(
                self._ocr_output,
            )

        for doc_id, image in tqdm.tqdm(input):
            ocr_results = ocr_task.process(doc_id, image)
            if ocr_results is None:
                continue

            results.append(ocr_results)

        return results
