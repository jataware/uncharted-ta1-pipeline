from flask import Flask, request, Response
import logging, json
from pathlib import Path
from hashlib import sha1

import io
from PIL import Image

import env_defaults as env

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

        # convert result to a JSON array
        result = results["doc_text_extraction_output"]
        if type(result) == BaseModelOutput:
            result_json = json.dumps(result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No text extracted"
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
    logger = logging.getLogger("text_extractor app")
    logger.info("*** Starting Text Extractor App ***")

    pipeline = TextExtractionPipeline(Path("/tmp/lara/workdir"), tile=True)

    #### start flask server
    app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
