from flask import Flask, request, Response
import logging, json
from PIL import Image
import argparse
from hashlib import sha1
from io import BytesIO

from pipelines.point_extraction.point_extraction_pipeline import PointExtractionPipeline
from tasks.common.pipeline import PipelineInput, BaseModelOutput


#
# Flask web app for Legend and Map Segmenter module
#

app = Flask(__name__)


@app.route("/api/process_image", methods=["POST"])
def process_image():
    """
    Perform point extraction on a map image.
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
        result = point_extraction_pipeline.run(pipeline_input)
        if len(result) == 0:
            msg = "No point extraction results"
            logging.warning(msg)
            return (msg, 500)

        point_extraction_result = result["map_point_label_output"]
        if isinstance(point_extraction_result, BaseModelOutput):
            # convert result to a JSON array
            result_json = json.dumps(point_extraction_result.data.model_dump())
            return Response(result_json, status=200, mimetype="application/json")
        else:
            msg = "No point extraction results"
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
    logger = logging.getLogger("point extraction app")
    logger.info("*** Starting Point Extraction App ***")

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_segmenter", type=str, default=None)
    parser.add_argument("--debug", type=float, default=False)
    p = parser.parse_args()

    # init segmenter
    point_extraction_pipeline = PointExtractionPipeline(
        p.model, p.model_segmenter, p.workdir
    )

    #### start flask server
    if p.debug:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=5000)

    # TEMP Use this for debug mode
