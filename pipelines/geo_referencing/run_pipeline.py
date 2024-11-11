import argparse
import logging
import os
from PIL.Image import Image as PILIMAGE
from PIL import Image

from pipelines.geo_referencing.georeferencing_pipeline import GeoreferencingPipeline
from pipelines.geo_referencing.pipeline_input_utils import (
    parse_query_file,
    get_geofence_defaults,
)
from pipelines.geo_referencing.output import CSVWriter, JSONWriter
from tasks.common.io import (
    BytesIOFileWriter,
    ImageFileInputIterator,
    JSONFileWriter,
)
from tasks.common.pipeline import (
    BaseModelOutput,
    BytesOutput,
    EmptyOutput,
    PipelineInput,
)
from tasks.geo_referencing.entities import (
    LEVERS_OUTPUT_KEY,
    PROJECTED_MAP_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    SCORING_OUTPUT_KEY,
    SUMMARY_OUTPUT_KEY,
    GEOREFERENCING_OUTPUT_KEY,
)

from tasks.metadata_extraction.metadata_extraction import LLM, LLM_PROVIDER
from util import logging as logging_util

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
        default="data/state_plane_reference.csv",
    )
    parser.add_argument(
        "--state_plane_zone_filename",
        type=str,
        default="data/USA_State_Plane_Zones_NAD27.geojson",
    )
    parser.add_argument(
        "--state_code_filename",
        type=str,
        default="data/state_codes.csv",
    )
    parser.add_argument(
        "--country_code_filename",
        type=str,
        default="data/country_codes.csv",
    )
    parser.add_argument(
        "--geocoded_places_filename",
        type=str,
        default="data/geocoded_places_reference.json",
    )
    parser.add_argument(
        "--ocr_gamma_correction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--llm", type=LLM, choices=list(LLM), default=LLM.GPT_4_O)
    parser.add_argument(
        "--llm_provider",
        type=LLM_PROVIDER,
        choices=list(LLM_PROVIDER),
        default=LLM_PROVIDER.OPENAI,
    )
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    p = parser.parse_args()

    # setup an input stream
    input = ImageFileInputIterator(p.input)

    run_pipeline(p, input)


def create_input(
    raster_id: str,
    image: PILIMAGE,
    query_path: str,
    geofence_region: str = "world",
) -> PipelineInput:
    input = PipelineInput()
    input.image = image
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_geofence_defaults(geofence_region)
    input.params["lon_minmax"] = lon_minmax
    input.params["lat_minmax"] = lat_minmax
    input.params["lon_sign_factor"] = lon_sign_factor

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
        parsed.geocoded_places_filename,
        parsed.ocr_gamma_correction,
        parsed.llm,
        parsed.llm_provider,
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
        elif isinstance(output_data, EmptyOutput):
            logger.info(
                f"no georeferencing results for {raster_id}, skipping writing to file"
            )

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
            elif isinstance(map_output, EmptyOutput):
                logger.info(
                    f"no projected map for {raster_id}, skipping writing to file"
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


if __name__ == "__main__":
    main()
