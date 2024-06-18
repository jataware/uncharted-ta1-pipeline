import argparse
from pathlib import Path
import logging
import os

from PIL.Image import Image as PILImage

from tasks.common.pipeline import PipelineInput, BaseModelOutput, ImageOutput
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.io import ImageFileInputIterator, ImageFileWriter, JSONFileWriter
from tasks.metadata_extraction.metadata_extraction import LLM
from util import logging as logging_util


def main():
    logger = logging.getLogger("metadata_pipeline")
    logging_util.config_logger(logger)

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cdr_schema", action="store_true")
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--llm", type=LLM, choices=list(LLM), default=LLM.GPT_3_5_TURBO)
    parser.add_argument("--no_gpu", action="store_true")
    p = parser.parse_args()

    logger.info(f"Args: {p}")

    # setup an input stream
    input = ImageFileInputIterator(str(p.input))

    # setup output writers
    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    # create the pipeline
    pipeline = MetadataExtractorPipeline(
        p.workdir, p.model, p.debug_images, p.cdr_schema, p.llm, not p.no_gpu
    )

    # run the extraction pipeline
    for doc_id, image in input:
        image_input = PipelineInput(image=image, raster_id=doc_id)
        results = pipeline.run(image_input)

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):
                if output_type == "metadata_extraction_output":
                    path = os.path.join(p.output, f"{doc_id}_metadata_extraction.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "metadata_cdr_output" and p.cdr_schema:
                    path = os.path.join(
                        p.output, f"{doc_id}_metadata_extraction_cdr.json"
                    )
                    file_writer.process(path, output_data.data)
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_metadata_extraction.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)
            else:
                logger.warning(f"Unknown output type: {type(output_data)}")
                continue


if __name__ == "__main__":
    main()
