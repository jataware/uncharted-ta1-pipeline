import argparse
import logging
import os
from PIL.Image import Image as PILImage
from tasks.common.pipeline import (
    EmptyOutput,
    PipelineInput,
    BaseModelOutput,
    ImageOutput,
)
from pipelines.segmentation.segmentation_pipeline import SegmentationPipeline
from tasks.common.io import (
    ImageFileInputIterator,
    JSONFileWriter,
    ImageFileWriter,
    validate_s3_config,
)

from util import logging as logging_util


def main():

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.25)
    parser.add_argument("--cdr_schema", action="store_true")
    parser.add_argument("--debug_images", action="store_true")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    p = parser.parse_args()

    logger = logging.getLogger("segmentation_pipeline")
    logging_util.config_logger(logger, p.log_level)

    # validate any s3 path args up front
    validate_s3_config(p.input, p.workdir, "", p.output)

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # setup an output writer
    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    # create the pipeline
    pipeline = SegmentationPipeline(
        p.model,
        p.workdir,
        p.debug_images,
        p.cdr_schema,
        p.min_confidence,
        not p.no_gpu,
    )

    # run the extraction pipeline
    for doc_id, image in input:
        logger.info(f"Processing doc_id: {doc_id}")
        image_input = PipelineInput(image=image, raster_id=doc_id)

        # run the pipeline
        try:
            results = pipeline.run(image_input)
        except Exception as e:
            logger.exception(e)
            continue

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):
                if output_type == "map_segmentation_output":
                    path = os.path.join(p.output, f"{doc_id}_map_segmentation.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "map_segmentation_cdr_output" and p.cdr_schema:
                    path = os.path.join(p.output, f"{doc_id}_map_segmentation_cdr.json")
                    file_writer.process(path, output_data.data)
                else:
                    logger.warning(f"Unknown output type: {output_type}")
            elif isinstance(output_data, ImageOutput):
                # write out the image
                path = os.path.join(p.output, f"{doc_id}_map_segmentation.png")
                assert isinstance(output_data.data, PILImage)
                image_writer.process(path, output_data.data)
            elif isinstance(output_data, EmptyOutput):
                logger.info(f"Empty {output_type} output for {doc_id}")
            else:
                logger.warning(f"Unknown output type: {type(output_data)}")
                continue


if __name__ == "__main__":
    main()
