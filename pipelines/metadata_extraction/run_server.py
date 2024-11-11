import argparse
from pathlib import Path
from flask import Flask, request, Response
import logging, json
from hashlib import sha1
from io import BytesIO

from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.request_client import (
    OutputType,
    RequestClient,
    METADATA_REQUEST_QUEUE,
    METADATA_RESULT_QUEUE,
)
from tasks.common.pipeline import (
    EmptyOutput,
    PipelineInput,
    BaseModelOutput,
    BaseModelListOutput,
)
from tasks.metadata_extraction.metadata_extraction import LLM, LLM_PROVIDER
from tasks.common import image_io
from tasks.metadata_extraction.entities import METADATA_EXTRACTION_OUTPUT_KEY
from util import logging as logging_util

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
        elif isinstance(metadata_result, EmptyOutput):
            msg = "No metadata extracted"
            logging.info(msg)
            return (msg, 200)
        else:
            msg = "No metadata extracted - unknown output type"
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
    logger = logging.getLogger("metadata extraction app")
    logging_util.config_logger(logger)

    logger.info("*** Starting map metadata app ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--imagedir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cdr_schema",
        action="store_true",
        help="Output results as TA1 json schema format",
    )
    parser.add_argument("--llm", type=LLM, choices=list(LLM), default=LLM.GPT_4_O)
    parser.add_argument(
        "--llm_provider",
        type=LLM_PROVIDER,
        choices=list(LLM_PROVIDER),
        default=LLM_PROVIDER.OPENAI,
    )
    parser.add_argument("--rest", action="store_true")
    parser.add_argument("--rabbit_host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--metrics_url", type=str, default="")
    parser.add_argument("--request_queue", type=str, default=METADATA_REQUEST_QUEUE)
    parser.add_argument("--result_queue", type=str, default=METADATA_RESULT_QUEUE)
    parser.add_argument("--no_gpu", action="store_true")
    p = parser.parse_args()

    # init segmenter
    metadata_extraction = MetadataExtractorPipeline(
        p.workdir,
        p.model,
        cdr_schema=p.cdr_schema,
        model=p.llm,
        gpu=not p.no_gpu,
        metrics_url=p.metrics_url,
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
        client = RequestClient(
            metadata_extraction,
            p.request_queue,
            p.result_queue,
            metadata_result_key,
            OutputType.METADATA,
            p.imagedir,
            host=p.rabbit_host,
            port=p.rabbit_port,
            vhost=p.rabbit_vhost,
            uid=p.rabbit_uid,
            pwd=p.rabbit_pwd,
            metrics_url=p.metrics_url,
            metrics_type="metadata",
        )
        client.start_request_queue()
        client.start_result_queue()
