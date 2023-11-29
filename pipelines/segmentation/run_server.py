from flask import Flask, request, Response
import logging, json
import numpy as np
from PIL import Image
import env_defaults as env
from hashlib import sha1
from io import BytesIO

from pipelines.segmentation.segmentation_pipeline import SegmentationPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput


#
# Flask web app for Legend and Map Segmenter module
#

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Perform legend and map segmentation on an image
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
        result = segmentation_pipeline.run(pipeline_input)
        if len(result) == 0:
            msg = "No segmentation results"
            logging.warning(msg)
            return (msg, 500)

        segmentation_result = result["map_segmenation_output"]
        if isinstance(segmentation_result, BaseModelOutput):
            # convert result to a JSON array
            result_json = json.dumps(segmentation_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No map segmentation results"
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
    logger = logging.getLogger("segmenter app")
    logger.info("*** Starting Legend and Map Segmenter App ***")

    # init segmenter
    segmentation_pipeline = SegmentationPipeline(
        str(env.MODEL_DATA_PATH),
        str(env.MODEL_DATA_CACHE_PATH),
        confidence_thres=env.MODEL_CONFIDENCE_THRES,
    )

    #### start flask server
    app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
