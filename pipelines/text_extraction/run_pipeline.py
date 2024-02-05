import argparse
import os
from tasks.common.io import ImageFileInputIterator, JSONFileWriter, ImageFileWriter
from tasks.common.pipeline import (
    PipelineInput,
    BaseModelOutput,
    BaseModelListOutput,
    ImageOutput,
)
from .text_extraction_pipeline import TextExtractionPipeline
from PIL.Image import Image as PILImage


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument(
        "--ta1_schema", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--no-tile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--pixel_limit", type=int, default=6000)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # run the extraction pipeline
    pipeline = TextExtractionPipeline(p.workdir, not p.no_tile, p.pixel_limit, p.debug)

    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

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
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_text_extraction.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)
            else:
                continue


if __name__ == "__main__":
    main()
