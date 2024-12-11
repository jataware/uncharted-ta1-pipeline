import argparse
from attr import validate
from flask import Flask, request, Response
import logging, json
from hashlib import sha1

from io import BytesIO
from tasks.common import image_io

from tasks.common.io import validate_s3_config
from tasks.common.request_client import (
    TEXT_REQUEST_QUEUE,
    TEXT_RESULT_QUEUE,
    RequestClient,
    OutputType,
)

from .text_extraction_pipeline import TextExtractionPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput
from util import logging as logging_util

#
# Flask web app for text extraction
#

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Perform ocr text extraction on an image
    request.data is expected to contain binary image file buffer
    """

    try:
        # open the image from the supplied byte stream
        bytes_io = BytesIO(request.data)
        image = image_io.load_pil_image_stream(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        input = PipelineInput(image=image, raster_id=doc_id)

        results = pipeline.run(input)
        if len(results) == 0:
            msg = "No text extracted"
            logging.warning(msg)
            return (msg, 500)

        result_key = app.config.get("result_key", "doc_text_extraction_output")
        text_result = results[result_key]

        if type(text_result) == BaseModelOutput:
            result_json = json.dumps(text_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No text extracted"
            logging.warning(msg)
            return (msg, 500)

    except Exception as e:
        msg = f"Error with process_image: {repr(e)}"
        logging.error(msg)
        logging.exception(e)
        return Response(msg, status=500)


@app.route("/healthcheck")
def health():
    """
    healthcheck
    """
    return ("healthy", 200)


if __name__ == "__main__":
    logger = logging.getLogger("text_extractor app")
    logging_util.config_logger(logger)
    logger.info("*** Starting Text Extractor App ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--imagedir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cdr_schema", action="store_true")
    parser.add_argument("--tile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel_limit", type=int, default=6000)
    parser.add_argument("--gamma_corr", type=float, default=1.0)
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--rabbit_host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--metrics_url", type=str, default="")
    parser.add_argument("--request_queue", type=str, default=TEXT_REQUEST_QUEUE)
    parser.add_argument("--result_queue", type=str, default=TEXT_RESULT_QUEUE)
    p = parser.parse_args()

    # validate s3 path args up front
    validate_s3_config("", p.workdir, p.imagedir, "")

    pipeline = TextExtractionPipeline(
        p.workdir, p.tile, p.pixel_limit, p.gamma_corr, p.debug, p.metrics_url
    )

    result_key = (
        "doc_text_extracction_output"
        if not p.cdr_schema
        else "doc_text_extraction_cdr_output"
    )
    app.config["result_key"] = result_key

    #### start flask server or startup up the message queue
    if p.rest:
        if p.debug:
            app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    else:
        client = RequestClient(
            pipeline,
            p.request_queue,
            p.result_queue,
            result_key,
            OutputType.TEXT,
            p.imagedir,
            host=p.rabbit_host,
            port=p.rabbit_port,
            vhost=p.rabbit_vhost,
            uid=p.rabbit_uid,
            pwd=p.rabbit_pwd,
            metrics_url=p.metrics_url,
            metrics_type="text",
        )
        client.start_request_queue()
        client.start_result_queue()
