from flask import Flask, request, Response
import logging, json
from pathlib import Path
import numpy as np
from hashlib import sha1
from typing import Tuple

# import cv2
# from urllib.parse import urlparse
import io
from PIL import Image

import env_defaults as env

from .text_extraction_pipeline import TextExtractionPipeline
from tasks.text_extraction.entities import DocTextExtraction


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

        input = iter([(doc_id, im)])

        results = pipeline.run(input)
        if len(results) == 0:
            msg = "No text extracted"
            logging.warning(msg)
            return (msg, 500)

        # convert result to a JSON array
        result_json = json.dumps(results[0].model_dump)

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("text_extractor app")
    logger.info("*** Starting Text Extractor App ***")

    pipeline = TextExtractionPipeline(
        Path("/tmp/lara/workdir"), tile=True, verbose=False
    )

    #### start flask server
    app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
