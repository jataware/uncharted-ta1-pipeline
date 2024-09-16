import argparse
import json
import logging
import os
from pathlib import Path

from flask import Flask, request, Response
from hashlib import sha1
from io import BytesIO
from PIL.Image import Image as PILImage
from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipeline
from pipelines.geo_referencing.output import (
    LARAModelOutput,
)
from tasks.common.pipeline import BaseModelOutput, Pipeline, PipelineInput
from tasks.common.queue import (
    GEO_REFERENCE_REQUEST_QUEUE,
    GEO_REFERENCE_RESULT_QUEUE,
    RequestQueue,
    OutputType,
)
from typing import Tuple
from util import logging as logging_util

Image.MAX_IMAGE_PIXELS = 400000000

app = Flask(__name__)

georef_pipeline: Pipeline


def get_geofence(
    lon_limits: Tuple[float, float] = (-66.0, -180.0),
    lat_limits: Tuple[float, float] = (24.0, 73.0),
    use_abs: bool = True,
):
    lon_minmax = lon_limits
    lat_minmax = lat_limits
    lon_sign_factor = 1.0

    if (
        use_abs
    ):  # use abs of lat/lon geo-fence? (since parsed OCR values don't usually include sign)
        if lon_minmax[0] < 0.0:
            lon_sign_factor = (
                -1.0
            )  # to account for -ve longitude values being forced to abs
            # (used when finalizing lat/lon results for query points)
        lon_minmax = [abs(x) for x in lon_minmax]
        lat_minmax = [abs(x) for x in lat_minmax]

        lon_minmax = [min(lon_minmax), max(lon_minmax)]
        lat_minmax = [min(lat_minmax), max(lat_minmax)]

    return (lon_minmax, lat_minmax, lon_sign_factor)


def create_input(raster_id: str, image: PILImage) -> PipelineInput:
    input = PipelineInput()
    input.image = image
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_geofence()
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
        image = Image.open(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the metadata extraction pipeline
        input = create_input(doc_id, image)
        outputs = georef_pipeline.run(input)
        if len(outputs) == 0:
            msg = "No georeferencing information derived"
            logging.warning(msg)
            return (msg, 500)

        result = outputs["georef_output"]
        if isinstance(result, BaseModelOutput):
            result_json = json.dumps(result.data.model_dump())
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
    logger = logging.getLogger("georef app")
    logging_util.config_logger(logger)

    logger.info("*** Starting geo referencing app ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default="tmp/lara/workdir")
    parser.add_argument("--imagedir", type=Path, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.25)
    parser.add_argument("--debug", type=float, default=False)
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--rabbit_host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument(
        "--request_queue", type=str, default=GEO_REFERENCE_REQUEST_QUEUE
    )
    parser.add_argument("--result_queue", type=str, default=GEO_REFERENCE_RESULT_QUEUE)
    parser.add_argument(
        "--country_code_filename",
        type=str,
        default="./data/country_codes.csv",
    )
    # TODO: DEFAULT VALUES SHOULD POINT TO PROPER FOLDER IN CONTAINER!
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
        "--ocr_gamma_correction",
        type=float,
        default=0.5,
    )
    parser.add_argument("--no_gpu", action="store_true")
    p = parser.parse_args()

    global georef_pipeline
    georef_pipeline = create_geo_referencing_pipeline(
        p.model,
        [LARAModelOutput("georef_output")],
        p.workdir,
        p.state_plane_lookup_filename,
        p.state_plane_zone_filename,
        p.state_code_filename,
        p.country_code_filename,
        p.ocr_gamma_correction,
        not p.no_gpu,
    )

    #### start flask server or startup up the message queue
    if p.rest:
        if p.debug:
            app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    else:
        queue = RequestQueue(
            georef_pipeline,
            p.request_queue,
            p.result_queue,
            "georef_output",
            OutputType.GEOREFERENCING,
            p.workdir,
            p.imagedir,
            host=p.rabbit_host,
            port=p.rabbit_port,
            vhost=p.rabbit_vhost,
            uid=p.rabbit_uid,
            pwd=p.rabbit_pwd,
        )
        queue.start_request_queue()
        queue.start_result_queue()


if __name__ == "__main__":
    start_server()
