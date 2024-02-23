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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("metadata_pipeline")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ta1_schema", type=bool, default=False)
    parser.add_argument("--debug_images", type=bool, default=False)
    parser.add_argument("--llm", type=LLM, choices=list(LLM), default=LLM.GPT_3_5_TURBO)
    parser.add_argument("--no_gpu", type=bool, default=False)
    p = parser.parse_args()

    logger.info(f"Args: {p}")

    # setup an input stream
    input = ImageFileInputIterator(str(p.input))

    # setup output writers
    file_writer = JSONFileWriter()
    image_writer = ImageFileWriter()

    # create the pipeline
    pipeline = MetadataExtractorPipeline(
        p.workdir, p.model, p.output, p.debug_images, p.ta1_schema, p.llm, not p.no_gpu
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
                elif output_type == "metadata_integration_output" and p.ta1_schema:
                    path = os.path.join(
                        p.output, f"{doc_id}_metadata_extraction_schema.json"
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
