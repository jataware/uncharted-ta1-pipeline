import argparse
import re
from flask import Flask, request, Response
import logging, json
from hashlib import sha1
from io import BytesIO

from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.queue import RequestQueue
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

        cdr_schema = app.config.get("cdr_schema", False)
        # get ta1 schema output or internal output format
        metadata_result = (
            result[METADATA_EXTRACTION_OUTPUT_KEY]
            if not cdr_schema
            else result["metadata_cdr_output"]
        )

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
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cdr_schema",
        action="store_true",
        help="Output results as TA1 json schema format",
    )
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--request_queue", type=str, default="metadata_request")
    parser.add_argument("--result_queue", type=str, default="metadata_result")
    p = parser.parse_args()

    # init segmenter
    metadata_extraction = MetadataExtractorPipeline(
        p.workdir, p.model, cdr_schema=p.cdr_schema
    )

    #### start flask server or startup up the message queue
    if p.rest:
        app.config["cdr_schema"] = p.cdr_schema
        if p.debug:
            app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
        else:
            app.run(host="0.0.0.0", port=5000)
    else:
        queue = RequestQueue(
            metadata_extraction,
            p.request_queue,
            p.result_queue,
            METADATA_EXTRACTION_OUTPUT_KEY,
            p.workdir,
        )
        queue.start_request_queue()
