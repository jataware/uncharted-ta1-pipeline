import argparse
import logging, os
from tasks.common.io import (
    ImageFileInputIterator,
    JSONFileWriter,
    ImageFileWriter,
    validate_s3_config,
)
from tasks.common.pipeline import (
    PipelineInput,
    BaseModelOutput,
    ImageOutput,
)
from .text_extraction_pipeline import TextExtractionPipeline
from PIL.Image import Image as PILImage
from util import logging as logging_util


def main():
    logger = logging.getLogger("text_extraction_pipeline")
    logging_util.config_logger(logger)

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument(
        "--cdr_schema", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--tile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel_limit", type=int, default=6000)
    parser.add_argument("--gamma_corr", type=float, default=1.0)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)

    p = parser.parse_args()

    # validate any s3 path args up front
    validate_s3_config(p.input, p.workdir, "", p.output)

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # run the extraction pipeline
    pipeline = TextExtractionPipeline(
        p.workdir, p.tile, p.pixel_limit, p.gamma_corr, p.debug
    )

    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    for doc_id, image in input:
        # run the model
        try:
            results = pipeline.run(PipelineInput(raster_id=doc_id, image=image))
        except Exception as e:
            logger.exception(e)
            continue

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):  # type assertion
                if output_type == "doc_text_extraction_output":
                    path = os.path.join(p.output, f"{doc_id}_text_extraction.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "doc_text_extraction_cdr_output" and p.cdr_schema:
                    path = os.path.join(
                        p.output, f"{doc_id}_text_extraction_schema.json"
                    )
                    file_writer.process(path, output_data.data)
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_text_extraction.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)


if __name__ == "__main__":
    main()
