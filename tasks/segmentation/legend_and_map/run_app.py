from flask import Flask, request, Response
import logging, json
import numpy as np
import cv2

from detectron_segmenter import DetectronSegmenter
import env_vars as env

#
# Flask web app for Legend and Map Segmenter module
#

app = Flask(__name__)


@app.route('/api/process_image', methods=['POST'])
def process_image():
    '''
    Perform legend and map segmentation on an image
    Image buffer is expected to be an opencv-formatted image object (numpy array) 
    '''

    # Adapted from code samples here: https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
    try:
        # convert string of image data to uint8
        nparr = np.frombuffer(request.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = segmenter.run_inference(img)

        # convert result to a JSON array
        result_json = json.dumps([dict(res) for res in results])

        return Response(result_json, status=200, mimetype="application/json")

    except Exception as e:
        msg = f'Error with process_image: {repr(e)}'
        logging.error(msg)
        print(repr(e)) 
        return Response(msg, status=500)
    

@app.route("/healthcheck")
def health():
    '''
    healthcheck
    '''
    return ("healthy", 200)


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s\t: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('segmenter app')
    logger.info('*** Starting Legend and Map Segmenter App ***')

    # init segmenter
    segmenter = DetectronSegmenter(env.MODEL_CONFIG_YML, env.MODEL_WEIGHTS, confidence_thres=env.MODEL_CONFIDENCE_THRES)

    #### start flask server
    app.run(host='0.0.0.0', port=5000)

    #TEMP Use this for debug mode
    #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)