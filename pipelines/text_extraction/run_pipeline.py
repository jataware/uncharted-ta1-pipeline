import argparse
import os
from pathlib import Path
from typing import List
from tasks.io.io import ImageFileInputIterator, JSONFileWriter
from tasks.text_extraction.text_extractor import SchemaTransformer
from .text_extraction_pipeline import TextExtractionPipeline


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ta1_schema", action="store_true")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--pixel_limit", type=int, default=6000)

    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # run the extraction pipeline
    pipeline = TextExtractionPipeline(p.workdir, p.tile, p.pixel_limit, p.verbose)
    results = pipeline.run(input)

    # write the results as TA1 schema Map files
    file_writer = JSONFileWriter()
    if p.ta1_schema:
        for result in results:
            path = os.path.join(p.output, f"{result.doc_id}_schema.json")
            tx_result = SchemaTransformer().process(result)
            file_writer.process(path, tx_result)
    else:
        for result in results:
            path = os.path.join(p.output, f"{result.doc_id}.json")
            file_writer.process(path, result)


if __name__ == "__main__":
    main()
