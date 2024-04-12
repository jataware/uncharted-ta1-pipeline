import argparse
from pathlib import Path
import logging
import os

from tasks.common.pipeline import PipelineInput, BaseModelOutput
from pipelines.point_extraction.point_extraction_pipeline import PointExtractionPipeline
from tasks.common.io import ImageFileInputIterator, JSONFileWriter


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("point_extraction_pipeline")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model_point_extractor", type=str, required=True)
    parser.add_argument("--model_segmenter", type=str, default=None)
    parser.add_argument("--cdr_schema", action="store_true")
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # setup an output writer
    file_writer = JSONFileWriter()

    # create the pipeline
    pipeline = PointExtractionPipeline(
        p.model_point_extractor, p.model_segmenter, p.workdir
    )

    # run the extraction pipeline
    for doc_id, image in input:
        image_input = PipelineInput(image=image, raster_id=doc_id)
        results = pipeline.run(image_input)

        # write the results out to the file system or s3 bucket
        for output_type, output_data in results.items():
            if isinstance(output_data, BaseModelOutput):
                if output_type == "map_point_label_output":
                    path = os.path.join(p.output, f"{doc_id}_point_extraction.json")
                    file_writer.process(path, output_data.data)
                elif output_type == "map_point_label_cdr_output" and p.cdr_schema:
                    path = os.path.join(p.output, f"{doc_id}_point_extraction_cdr.json")
                    file_writer.process(path, output_data.data)
            else:
                logger.warning(f"Unknown output data: {output_data}")


if __name__ == "__main__":
    main()
