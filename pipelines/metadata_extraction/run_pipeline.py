import argparse
from pathlib import Path
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.io.io import ImageFileInputIterator
from tasks.metadata_extraction.metadata_extraction import SchemaFileWriter


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # run the extraction pipeline
    pipeline = MetadataExtractorPipeline(p.workdir)
    results = pipeline.run(input)

    # write the results as TA1 schema Map files
    schema_file_writer = SchemaFileWriter(p.output)
    for result in results:
        schema_file_writer.process(result)


if __name__ == "__main__":
    main()
