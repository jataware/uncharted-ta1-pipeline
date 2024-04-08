import argparse
from flask import Flask, request, Response
import logging, json
from hashlib import sha1

import io
from PIL import Image

from .text_extraction_pipeline import TextExtractionPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput

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
        # decode and open as a PIL Image
        im = Image.open(io.BytesIO(request.data))

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        input = PipelineInput(image=im, raster_id=doc_id)

        results = pipeline.run(input)
        if len(results) == 0:
            msg = "No text extracted"
            logging.warning(msg)
            return (msg, 500)

        cdr_schema = app.config.get("cdr_schema", False)
        text_result = (
            results["doc_text_extraction_cdr_output"]
            if cdr_schema
            else results["doc_text_extraction_output"]
        )

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
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("text_extractor app")
    logger.info("*** Starting Text Extractor App ***")

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="tmp/lara/workdir")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cdr_schema", action="store_true")
    p = parser.parse_args()

    pipeline = TextExtractionPipeline(p.workdir, tile=True)

    #### start flask server
    app.config["cdr_schema"] = p.cdr_schema
    if p.debug:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=5000)
