from pathlib import Path
import os
from . import ocr_util
from .entities import DocTextExtraction, TextExtraction, Point
from typing import List, Dict, Any
from PIL.Image import Image


class GoogleVisionOCR:
    def __init__(self, cache_path: Path, blocks: bool, document_ocr: bool):
        self._blocks = blocks
        self._document_ocr = document_ocr
        self._cache_path = cache_path

    def process(self, doc_id: str, input: Image) -> DocTextExtraction:
        """Runs OCR on and writes the output to a json file"""
        # if the output dir doesn't exist, create it
        if not self._cache_path.exists():
            os.makedirs(self._cache_path)

        texts = ocr_util.process_image(
            doc_id, input, self._cache_path, self._blocks, self._document_ocr
        )
        return self._from_legacy_text(doc_id, texts)

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
