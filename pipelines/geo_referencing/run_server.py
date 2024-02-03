import argparse
import logging
import os

from flask import Flask, request, Response
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from PIL.Image import Image as PILImage
from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipeline
from pipelines.geo_referencing.output import JSONWriter, ObjectOutput
from tasks.common.pipeline import Pipeline, PipelineInput
from tasks.geo_referencing.georeference import QueryPoint

from typing import Dict, List, Tuple

Image.MAX_IMAGE_PIXELS = 400000000


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/credentials.json"

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
        output_schema: ObjectOutput = outputs["schema"]  # type: ignore
        writer_json = JSONWriter()
        result_json = writer_json.output([output_schema], {})
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
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("georef app")
    logger.info("*** Starting geo referencing app ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.25)
    parser.add_argument("--debug", type=float, default=False)
    p = parser.parse_args()

    global georef_pipeline
    georef_pipeline = create_geo_referencing_pipeline(p.model)

    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    start_server()
