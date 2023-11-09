import os
from pathlib import Path
import tqdm
from tasks.metadata_extraction.metadata_extraction import (
    MetadataExtractor,
)
from tasks.text_extraction_2.gva_ocr import GoogleVisionOCR
from tasks.metadata_extraction.entities import MetadataExtraction
from typing import Iterator, Tuple, List
from PIL.Image import Image as PILImage


class MetadataExtractorPipeline:
    def __init__(
        self,
        work_dir: Path,
        verbose=False,
    ):
        self._ocr_output = Path(os.path.join(work_dir, "ocr_output"))
        self._verbose = verbose

    def run(self, input: Iterator[Tuple[str, PILImage]]) -> List[MetadataExtraction]:
        # instantiate the loader
        ocr_task = GoogleVisionOCR(self._ocr_output, True, True)
        metadata_extractor = MetadataExtractor(self._verbose)

        result: List[MetadataExtraction] = []
        for doc_id, image in tqdm.tqdm(input):
            print(f"Processing image: {doc_id}")

            # run google vision ocr
            ocr_results = ocr_task.process(doc_id, image)
            if ocr_results is None:
                continue

            # run metadata extraction
            metadata = metadata_extractor.process(ocr_results)
            if metadata is None:
                continue

            result.append(metadata)
        return result
