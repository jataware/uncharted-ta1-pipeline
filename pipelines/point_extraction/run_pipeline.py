import argparse
import json
import logging
import os

from tasks.common.pipeline import PipelineInput, BaseModelOutput
from pipelines.point_extraction.point_extraction_pipeline import PointExtractionPipeline
from tasks.common.io import ImageFileInputIterator, JSONFileWriter
from tasks.point_extraction.entities import (
    LegendPointItems,
    LEGEND_ITEMS_OUTPUT_KEY,
)


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
    parser.add_argument("--legend_hints_dir", type=str, default="")
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    # setup an output writer
    file_writer = JSONFileWriter()

    # create the pipeline
    pipeline = PointExtractionPipeline(
        p.model_point_extractor,
        p.model_segmenter,
        p.workdir,
        p.cdr_schema,
    )

    # run the extraction pipeline
    for doc_id, image in input:

        # --- TEMP code needed to run with contest dir-based data
        if (
            doc_id.endswith("_pt")
            or doc_id.endswith("_poly")
            or doc_id.endswith("_line")
            or doc_id.endswith("_point")
        ):
            logger.info(f"Skipping {doc_id}")
            continue
        # ---
        logger.info(f"Processing {doc_id}")
        image_input = PipelineInput(image=image, raster_id=doc_id)

        # load JSON legend hints file, if present, parse and add to PipelineInput
        if p.legend_hints_dir:
            try:
                # check or legend hints for this image (JSON CMA contest data)
                with open(
                    os.path.join(p.legend_hints_dir, doc_id + ".json"), "r"
                ) as fp:
                    legend_hints = json.load(fp)
                    legend_pt_items = LegendPointItems.parse_legend_point_hints(
                        legend_hints
                    )
                    # add legend item hints as a pipeline input param
                    image_input.params[LEGEND_ITEMS_OUTPUT_KEY] = legend_pt_items
                    logger.info(
                        f"Number of legend point items loaded for this map: {len(legend_pt_items.items)}"
                    )

            except Exception as e:
                logger.error("EXCEPTION loading legend hints json: " + repr(e))

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
