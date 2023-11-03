from pathlib import Path
import os
from . import ocr_util
from .entities import DocTextExtraction, TextExtraction, Point
from typing import List, Dict, Any


class GoogleVisionOCR:
    def __init__(
        self, input: Path, output: Path, blocks: bool, document_ocr: bool, show: bool
    ):
        self._input = input
        self._output = output
        self._blocks = blocks
        self._document_ocr = document_ocr

    def process(self) -> List[DocTextExtraction]:
        """Runs OCR on a single image or a directory of images and writes the output to a json file"""
        # if the output dir doesn't exist, create it
        if not self._output.exists():
            os.makedirs(self._output)

        if self._input.is_file():
            doc_id, texts = ocr_util.process_image(
                self._input, self._output, self._blocks, self._document_ocr
            )
            return [self._from_legacy_text(doc_id, texts)]

        else:
            ocr_results = ocr_util.process_images(
                self._input, self._output, self._blocks, self._document_ocr
            )
            doc_extractions: List[DocTextExtraction] = []
            for doc_id, texts in ocr_results:
                doc_extractions.append(self._from_legacy_text(doc_id, texts))
            return doc_extractions

    def _from_legacy_text(
        self, doc_id: str, texts: List[Dict[str, Any]]
    ) -> DocTextExtraction:
        """Converts a list of legacy text objects to a list of TextExtraction objects"""
        text_extractions: List[TextExtraction] = []
        for text in texts:
            bounding_poly = text["bounding_poly"]
            bounds = [
                Point(x=vertex.x, y=vertex.y) for vertex in bounding_poly.vertices
            ]
            text_extraction = TextExtraction(
                text=text["text"], confidence=1.0, bounds=bounds
            )
            text_extractions.append(text_extraction)

        return DocTextExtraction(doc_id=doc_id, extractions=text_extractions)
