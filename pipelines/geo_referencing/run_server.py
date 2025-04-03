import argparse
import base64
import json
import logging

from flask import Flask, request, Response
from hashlib import sha1
from io import BytesIO
from PIL.Image import Image as PILImage
from PIL import Image

from pipelines.geo_referencing.georeferencing_pipeline import GeoreferencingPipeline
from pipelines.geo_referencing.pipeline_input_utils import get_geofence_defaults
from pipelines.geo_referencing.output import (
    GeoreferencingOutput,
    ProjectedMapOutput,
)
from tasks.common.io import validate_s3_config
from tasks.common.pipeline import (
    BaseModelOutput,
    BytesOutput,
    OutputCreator,
    PipelineInput,
)
from tasks.common.request_client import (
    GEO_REFERENCE_REQUEST_QUEUE,
    GEO_REFERENCE_RESULT_QUEUE,
    REQUEUE_LIMIT,
    RequestClient,
    OutputType,
)
from tasks.common import image_io
from typing import List, Tuple
from tasks.geo_referencing.entities import (
    GEOREFERENCING_OUTPUT_KEY,
    PROJECTED_MAP_OUTPUT_KEY,
)
from tasks.metadata_extraction.metadata_extraction import (
    DEFAULT_GPT_MODEL,
    DEFAULT_OPENAI_API_VERSION,
    LLM_PROVIDER,
)
from util import logging as logging_util

Image.MAX_IMAGE_PIXELS = 400000000

app = Flask(__name__)

georef_pipeline: GeoreferencingPipeline

logger = logging.getLogger("georef app")


def create_input(
    raster_id: str, image: PILImage, geofence_region: str = "world"
) -> PipelineInput:
    input = PipelineInput()
    input.image = image
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_geofence_defaults(geofence_region)
    input.params["lon_minmax"] = lon_minmax
    input.params["lat_minmax"] = lat_minmax
    input.params["lon_sign_factor"] = lon_sign_factor

    return input


@app.route("/api/process_image", methods=["POST"])
def process_image():
    # Adapted from code samples here: https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
    try:
        # open the image from the supplied byte stream
        bytes_io = BytesIO(request.data)
        image = image_io.load_pil_image_stream(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the metadata extraction pipeline
        input = create_input(doc_id, image)
        outputs = georef_pipeline.run(input)
        if len(outputs) == 0:
            msg = "No georeferencing information derived"
            logging.warning(msg)
            return (msg, 500)

        result_json = ""
        result_dict = {}
        if GEOREFERENCING_OUTPUT_KEY in outputs:
            result = outputs[GEOREFERENCING_OUTPUT_KEY]
            if isinstance(result, BaseModelOutput):
                result_dict = result.data.model_dump()

        # get the projected map if its present and convert to base64
        if PROJECTED_MAP_OUTPUT_KEY in outputs:
            map = outputs[PROJECTED_MAP_OUTPUT_KEY]
            if isinstance(map, BytesOutput):
                map_str = base64.b64encode(map.data.getvalue()).decode()
                result_dict["projected_map"] = map_str

            result_json = json.dumps(result_dict)

        else:
            msg = "No point extraction results"
            logging.warning(msg)
            return (msg, 500)

        return Response(result_json, status=200, mimetype="application/json")

    except Exception as e:
        msg = f"Error with process_image: {repr(e)}"
        logging.error(msg)
        print(repr(e))
        return Response(msg, status=500)


@app.route("/healthcheck")
def health():
    """
    healthcheck
    """
    return ("healthy", 200)


def start_server():

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--imagedir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.25)
    parser.add_argument("--debug", type=float, default=False)
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--rabbit_host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--metrics_url", type=str, default="")
    parser.add_argument("--requeue_limit", type=int, default=REQUEUE_LIMIT)
    parser.add_argument(
        "--request_queue", type=str, default=GEO_REFERENCE_REQUEST_QUEUE
    )
    parser.add_argument("--result_queue", type=str, default=GEO_REFERENCE_RESULT_QUEUE)
    parser.add_argument(
        "--country_code_filename",
        type=str,
        default="data/country_codes.csv",
    )
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
        "--geocoded_places_filename",
        type=str,
        default="data/geocoded_places_reference.json",
    )
    parser.add_argument(
        "--ocr_gamma_correction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--llm", type=str, default=DEFAULT_GPT_MODEL)
    parser.add_argument(
        "--llm_api_version", type=str, default=DEFAULT_OPENAI_API_VERSION
    )
    parser.add_argument(
        "--llm_provider",
        type=LLM_PROVIDER,
        choices=list(LLM_PROVIDER),
        default=LLM_PROVIDER.OPENAI,
    )
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--project", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--ocr_cloud_auth", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    p = parser.parse_args()

    logging_util.config_logger(logger, p.log_level)
    logger.info("*** Starting geo referencing app ***")

    # validate any s3 path args up front
    validate_s3_config("", p.workdir, p.imagedir, "")

    outputs: List[OutputCreator] = [GeoreferencingOutput("georeferencing")]
    if p.project:
        if not p.rest:
            raise ValueError(
                "Projecting the map is only supported in REST mode, not in queue mode"
            )
        outputs.append(ProjectedMapOutput(PROJECTED_MAP_OUTPUT_KEY))

    global georef_pipeline
    georef_pipeline = GeoreferencingPipeline(
        p.workdir,
        p.model,
        p.state_plane_lookup_filename,
        p.state_plane_zone_filename,
        p.state_code_filename,
        p.country_code_filename,
        p.geocoded_places_filename,
        p.ocr_gamma_correction,
        p.llm,
        p.llm_api_version,
        p.llm_provider,
        p.project,
        p.diagnostics,
        not p.no_gpu,
        p.metrics_url,
        p.ocr_cloud_auth,
    )

    #### start flask server or startup up the message queue
    if p.rest:
        if p.debug:
            app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    else:
        client = RequestClient(
            georef_pipeline,
            p.request_queue,
            p.result_queue,
            GEOREFERENCING_OUTPUT_KEY,
            OutputType.GEOREFERENCING,
            p.imagedir,
            host=p.rabbit_host,
            port=p.rabbit_port,
            vhost=p.rabbit_vhost,
            uid=p.rabbit_uid,
            pwd=p.rabbit_pwd,
            metrics_url=p.metrics_url,
            metrics_type="georef",
            requeue_limit=p.requeue_limit,
        )
        client.start_request_queue()
        client.start_result_queue()


if __name__ == "__main__":
    start_server()
