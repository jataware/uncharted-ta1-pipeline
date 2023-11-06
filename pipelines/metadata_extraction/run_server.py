from flask import Flask, request, Response
import logging, json
import numpy as np
import cv2.cv2 as cv2
from urllib.parse import urlparse
from metadata_extraction_pipeline import MetadataExtractorPipeline, SchemaTransformer
from hashlib import sha1
from io import BytesIO
from PIL import Image

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
        image = Image.open(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # ran the image through the metadata extraction pipeline
        data = iter([(doc_id, image)])
        result = metadata_extraction.run(data)
        map_json = json.dumps(SchemaTransformer().transform(result))

        # convert result to a JSON array
        return Response(map_json, status=200, mimetype="application/json")

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
    logger = logging.getLogger("segmenter app")
    logger.info("*** Starting Legend and Map Segmenter App ***")

    # init segmenter
    metadata_extraction = MetadataExtractorPipeline("work_dir")

    #### start flask server
    app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
