import argparse
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
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--ta1_schema", type=bool, default=False)
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # run the extraction pipeline
    pipeline = TextExtractionPipeline(p.workdir, p.verbose)
    results = pipeline.run(input)

    # write the results as TA1 schema Map files
    file_writer = JSONFileWriter()
    if p.ta1_schema:
        for r in results:
            tx_result = SchemaTransformer().process(r)
            file_writer.process(p.output, tx_result)
    else:
        for result in results:
            file_writer.process(p.output, result)


if __name__ == "__main__":
    main()
