import os
from pathlib import Path
from tasks.metadata_extraction.metadata_extraction import (
    MetadataExtractor,
    MetadataFileWriter,
    TA1SchemaFileWriter,
)
from tasks.text_extraction_2.gva_ocr import GoogleVisionOCR


class MetadataExtractorPipeline:
    def __init__(
        self, input: Path, output: str, work_dir: Path, verbose=False, append=False
    ):
        self._pipeline_input = input
        self._ocr_output = Path(os.path.join(work_dir, "ocr_output"))
        self._metadata_input = self._ocr_output
        self._metadata_processed = Path(os.path.join(work_dir, "metadata_processed"))
        self._pipeline_output = output
        self._verbose = verbose

    def run(self):
        # run google vision ocr
        print(f"Running OCR on {self._pipeline_input}")
        ocr_task = GoogleVisionOCR(
            self._pipeline_input, self._ocr_output, True, True, False
        )
        ocr_results = ocr_task.process()

        # run metadata extraction
        print(f"Running metadata extraction on {self._ocr_output}")
        metadata_extractor = MetadataExtractor(
            ocr_results, self._pipeline_output, self._verbose
        )
        metadata = metadata_extractor.process()

        # serialize output
        print(f"Serializing output to {self._pipeline_output}")
        metadata_serializer = TA1SchemaFileWriter(metadata, str(self._pipeline_output))
        metadata_serializer.process()
