import argparse
from pathlib import Path
import logging
import os
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.metadata_extraction.metadata_extraction import SchemaTransformer
from tasks.io.io import ImageFileInputIterator, JSONFileWriter


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
    pipeline = MetadataExtractorPipeline(p.workdir)
    results = pipeline.run(input)

    # write the results as TA1 schema Map files
    file_writer = JSONFileWriter()
    for result in results:
        if p.ta1_schema:
            schema_result = SchemaTransformer().process(result)
            path = os.path.join(p.output, f"{result.map_id}_map_schema.json")
            file_writer.process(path, schema_result)
        else:
            path = os.path.join(p.output, f"{result.map_id}_metadata.json")
            file_writer.process(path, result)


if __name__ == "__main__":
    main()
