import argparse
import os
from pathlib import Path
from typing import List, Any, Dict
from common.io import ImageFileInputIterator, JSONFileWriter
from tasks.common.pipeline import PipelineInput, BaseModelOutput, BaseModelListOutput
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
    pipeline = TextExtractionPipeline(p.workdir, p.tile, p.pixel_limit)

    file_writer = JSONFileWriter()
    for doc_id, image in input:
        # run the model
        results = pipeline.run(PipelineInput(raster_id=doc_id, image=image))

        # write the results out to the file system or s3 bucket
        for _, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):  # type assertion
                path = os.path.join(p.output, f"{doc_id}_text_extraction.json")
                file_writer.process(path, output_data.data)
            elif (
                isinstance(output_data, BaseModelListOutput) and p.ta1_schema
            ):  # type assertion
                path = os.path.join(p.output, f"{doc_id}_text_extraction_schema.json")
                file_writer.process(path, output_data.data)
            else:
                continue


if __name__ == "__main__":
    main()
