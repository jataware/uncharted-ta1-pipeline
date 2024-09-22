import argparse
import logging
import os

from PIL.Image import Image as PILIMAGE
from PIL import Image

from pipelines.geo_referencing.georeferencing_pipeline import GeoreferencingPipeline
from pipelines.geo_referencing.output import CSVWriter, JSONWriter
from tasks.common.io import (
    BytesIOFileWriter,
    ImageFileInputIterator,
    JSONFileWriter,
)
from tasks.common.pipeline import BaseModelOutput, BytesOutput, PipelineInput
from tasks.geo_referencing.entities import (
    LEVERS_OUTPUT_KEY,
    PROJECTED_MAP_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    SCORING_OUTPUT_KEY,
    SUMMARY_OUTPUT_KEY,
    GEOREFERENCING_OUTPUT_KEY,
)
from tasks.geo_referencing.georeference import QueryPoint
from util import logging as logging_util
from typing import List, Optional, Tuple

IMG_FILE_EXT = "tif"

Image.MAX_IMAGE_PIXELS = 400000000
GEOCODE_CACHE = "temp/geocode/"

logger = logging.getLogger("georeferencing_pipeline")
logging_util.config_logger(logger)


def main():

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--query_dir", type=str, default="")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--state_plane_lookup_filename",
        type=str,
        default="./data/state_plane_reference.csv",
    )
    parser.add_argument(
        "--state_plane_zone_filename",
        type=str,
        default="./data/USA_State_Plane_Zones_NAD27.geojson",
    )
    parser.add_argument(
        "--state_code_filename",
        type=str,
        default="./data/state_codes.csv",
    )
    parser.add_argument(
        "--country_code_filename",
        type=str,
        default="./data/country_codes.csv",
    )
    parser.add_argument(
        "--ocr_gamma_correction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    run_pipeline(p, input)


def create_input(raster_id: str, image: PILIMAGE, query_path: str) -> PipelineInput:
    input = PipelineInput()
    input.image = image
    input.raster_id = raster_id

    # if a query path is specified, parse the query file and the contents to the
    # query points output key for consumption within the pipeline
    if query_path != "":
        query_pts = parse_query_file(query_path, input.image.size)
        input.params[QUERY_POINTS_OUTPUT_KEY] = query_pts

    return input


def run_pipeline(parsed, input_data: ImageFileInputIterator):
    assert logger is not None

    pipeline = GeoreferencingPipeline(
        parsed.workdir,
        parsed.model,
        parsed.state_plane_lookup_filename,
        parsed.state_plane_zone_filename,
        parsed.state_code_filename,
        parsed.country_code_filename,
        parsed.ocr_gamma_correction,
        parsed.project,
        parsed.diagnostics,
        not parsed.no_gpu,
    )

    # get file paths
    query_dir = parsed.query_dir

    writer_csv = CSVWriter()
    writer_json = JSONWriter()
    writer_bytes = BytesIOFileWriter()
    writer_json_file = JSONFileWriter()

    results_scoring = []
    results_summary = []
    results_levers = []
    results_gcps = []

    for raster_id, image in input_data:
        logger.info(f"processing {raster_id}")

        query_path = (
            os.path.join(query_dir, raster_id + ".csv") if query_dir != "" else ""
        )

        input = create_input(raster_id, image, query_path)

        logger.info(f"running pipeline {pipeline.id}")
        output = pipeline.run(input)
        logger.info(f"done pipeline {pipeline.id}\n\n")

        # store the baseline georeferencing results
        output_data = output[GEOREFERENCING_OUTPUT_KEY]
        if isinstance(output_data, BaseModelOutput):
            path = os.path.join(parsed.output, f"{raster_id}_georeferencing.json")
            writer_json_file.process(path, output_data.data)

        # immediately write projected map to file - these are large so we don't want to accumulate them
        # in memory like the other results
        if parsed.project and PROJECTED_MAP_OUTPUT_KEY in output:
            map_output = output[PROJECTED_MAP_OUTPUT_KEY]
            if isinstance(map_output, BytesOutput):
                if len(map_output.data.getbuffer()) == 0:
                    logger.warning(
                        f"projected map for {raster_id} is empty, skipping writing to file"
                    )
                    continue
                output_path = os.path.join(
                    parsed.output, f"{raster_id}_projected_map.tif"
                )
                logger.info(f"writing projected map to {output_path}")
                writer_bytes.process(
                    output_path,
                    map_output.data,
                )

        # store the diagnostic info if present
        if SCORING_OUTPUT_KEY in output:
            results_scoring.append(output[SCORING_OUTPUT_KEY])
        if SUMMARY_OUTPUT_KEY in output:
            results_summary.append(output[SUMMARY_OUTPUT_KEY])
        if LEVERS_OUTPUT_KEY in output:
            results_levers.append(output[LEVERS_OUTPUT_KEY])
        if QUERY_POINTS_OUTPUT_KEY in output:
            results_gcps.append(output[QUERY_POINTS_OUTPUT_KEY])

    # write out the diagnostic info if present
    writer_csv.output(
        results_scoring,
        {"path": os.path.join(parsed.output, f"score_{pipeline.id}.csv")},
    )
    writer_csv.output(
        results_summary,
        {"path": os.path.join(parsed.output, f"summary_{pipeline.id}.csv")},
    )
    writer_json.output(
        results_levers,
        {"path": os.path.join(parsed.output, f"levers_{pipeline.id}.json")},
    )
    writer_json.output(
        results_gcps,
        {"path": os.path.join(parsed.output, f"gcps_{pipeline.id}.json")},
    )


def parse_query_file(
    csv_query_file: str, image_size: Optional[Tuple[float, float]] = None
) -> List[QueryPoint]:
    """
    Expected schema is of the form:
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,8250,12796,-105.72065081057087,43.40255034572461
    ...
    Note: NAD83* columns may not be present
    row (y) and col (x) = pixel coordinates to query
    NAD83* = (if present) are ground truth answers (lon and lat) for the query x,y pt
    """

    first_line = True
    x_idx = 2
    y_idx = 1
    lon_idx = 3
    lat_idx = 4
    query_pts = []
    try:
        with open(csv_query_file) as f_in:
            for line in f_in:
                if line.startswith("raster_") or first_line:
                    first_line = False
                    continue  # header line, skip

                rec = line.split(",")
                if len(rec) < 3:
                    continue
                raster_id = rec[0]
                x = int(rec[x_idx])
                y = int(rec[y_idx])
                if image_size is not None:
                    # sanity check that query points are not > image dimensions!
                    if x > image_size[0] or y > image_size[1]:
                        err_msg = (
                            "Query point {}, {} is outside image dimensions".format(
                                x, y
                            )
                        )
                        raise IOError(err_msg)
                lonlat_gt = None
                if len(rec) >= 5:
                    lon = float(rec[lon_idx])
                    lat = float(rec[lat_idx])
                    if lon != 0 and lat != 0:
                        lonlat_gt = (lon, lat)
                query_pts.append(QueryPoint(raster_id, (x, y), lonlat_gt))

    except Exception as e:
        logger.exception(f"EXCEPTION parsing query file: {str(e)}", exc_info=True)

    return query_pts


if __name__ == "__main__":
    main()
