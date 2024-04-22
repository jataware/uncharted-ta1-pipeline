import argparse
from pathlib import Path
from unittest.mock import DEFAULT
from flask import Flask, request, Response
import logging, json
from hashlib import sha1
from io import BytesIO

from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.queue import (
    OutputType,
    RequestQueue,
    METADATA_REQUEST_QUEUE,
    METADATA_RESULT_QUEUE,
)
from tasks.common.pipeline import PipelineInput, BaseModelOutput, BaseModelListOutput
from tasks.common import image_io
from tasks.metadata_extraction.entities import METADATA_EXTRACTION_OUTPUT_KEY

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    performs metadata extraction on an image
    request.data is expected to contain binary image file buffer
    """

    # Adapted from code samples here: https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
    try:
        # open the image from the supplied byte stream
        bytes_io = BytesIO(request.data)
        image = image_io.load_pil_image_stream(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the metadata extraction pipeline
        pipeline_input = PipelineInput(image=image, raster_id=doc_id)
        result = metadata_extraction.run(pipeline_input)
        if len(result) == 0:
            msg = "No metadata extracted"
            logging.warning(msg)
            return (msg, 500)

        result_key = app.config.get("result_key", METADATA_EXTRACTION_OUTPUT_KEY)
        metadata_result = result[result_key]

        # convert result to a JSON string and return
        if isinstance(metadata_result, BaseModelOutput):
            result_json = json.dumps(metadata_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        elif isinstance(metadata_result, BaseModelListOutput):
            result_json = json.dumps([d.model_dump() for d in metadata_result.data])
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No metadata extracted"
            logging.warning(msg)
            return (msg, 500)

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("metadata extraction app")
    logger.info("*** Starting map metadata app ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default="tmp/lara/workdir")
    parser.add_argument("--imagedir", type=Path, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cdr_schema",
        action="store_true",
        help="Output results as TA1 json schema format",
    )
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--rabbit_host", type=str, default="localhost")
    parser.add_argument("--request_queue", type=str, default=METADATA_REQUEST_QUEUE)
    parser.add_argument("--result_queue", type=str, default=METADATA_RESULT_QUEUE)
    p = parser.parse_args()

    # init segmenter
    metadata_extraction = MetadataExtractorPipeline(
        p.workdir, p.model, cdr_schema=p.cdr_schema
    )

    metadata_result_key = (
        METADATA_EXTRACTION_OUTPUT_KEY if not p.cdr_schema else "metadata_cdr_output"
    )

    #### start flask server or startup up the message queue
    if p.rest:
        app.config["result_key"] = metadata_result_key
        if p.debug:
            app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    else:
        queue = RequestQueue(
            metadata_extraction,
            p.request_queue,
            p.result_queue,
            metadata_result_key,
            OutputType.METADATA,
            p.workdir,
            p.imagedir,
            host=p.rabbit_host,
        )
        queue.start_request_queue()
