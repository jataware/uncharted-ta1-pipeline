import os
from flask import Flask, request, Response
import logging, json
from hashlib import sha1
from pathlib import Path
from io import BytesIO
from PIL import Image
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)
from tasks.common.pipeline import PipelineInput, BaseModelOutput
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
        image = Image.open(bytes_io)

        # use the hash as the doc id since we don't have a filename
        doc_id = sha1(request.data).hexdigest()

        # run the image through the metadata extraction pipeline
        pipeline_input = PipelineInput(image=image, raster_id=doc_id)
        result = metadata_extraction.run(pipeline_input)
        if len(result) == 0:
            msg = "No metadata extracted"
            logging.warning(msg)
            return (msg, 500)
        metadata_result = result[METADATA_EXTRACTION_OUTPUT_KEY]
        if isinstance(metadata_result, BaseModelOutput):
            # convert result to a JSON array
            result_json = json.dumps(metadata_result.data.model_dump())
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

    # init segmenter
    metadata_extraction = MetadataExtractorPipeline(
        "tmp/lara/workdir",
        "tmp/lara/models",
    )

    #### start flask server
    app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
