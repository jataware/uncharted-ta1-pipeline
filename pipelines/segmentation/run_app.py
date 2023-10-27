from flask import Flask, request, Response
import logging, json
import numpy as np
import cv2
from s3_data_cache import S3DataCache
from urllib.parse import urlparse

import env_defaults as env
from detectron_segmenter import DetectronSegmenter


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

        results = segmenter.run_inference(img, ta1_schema_out=env.USE_TA1_SCHEMA)

        # convert result to a JSON array
        result_json = json.dumps([res.model_dump() for res in results])

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


def prepare_data_cache():
    '''
    prepare local data cache
    and download model weights, if needed
    '''

    # check if path is a URL
    is_url = env.MODEL_WEIGHTS_PATH.startswith('s3://') or env.MODEL_WEIGHTS_PATH.startswith('http')
    s3_host = ''
    s3_path = ''
    s3_bucket = ''

    if is_url:
        res = urlparse(env.MODEL_WEIGHTS_PATH)
        s3_host = res.scheme + '://' + res.netloc
        s3_path = res.path.lstrip('/')
        s3_bucket = s3_path.split('/')[0]
        s3_path = s3_path.lstrip(s3_bucket)
        s3_path = s3_path.lstrip('/')

    # create local data cache, if doesn't exist, and connect to S3
    s3_data_cache = S3DataCache(env.LOCAL_DATA_CACHE, s3_host, s3_bucket, aws_access_key_id=env.AWS_ACCESS_KEY_ID, aws_secret_access_key=env.AWS_SECRET_ACCESS_KEY)

    if is_url:
        # download S3 model data to local data cache, if necessary
        local_model_weights = s3_data_cache.fetch_file_from_s3(s3_path, overwrite=False)
        if local_model_weights:
            env.MODEL_WEIGHTS_PATH = local_model_weights    # use locally-cached model weights 

        # check for yml or json config files in the same sub-folder,
        # and download as needed
        s3_subfolder = s3_path[:s3_path.rfind('/')]         # get path up to last subfolder
        for s3_key in s3_data_cache.list_bucket_contents(s3_subfolder):
            if s3_key.endswith('.yaml') or s3_key.endswith('.yml') or s3_key.endswith('.json'):
                s3_data_cache.fetch_file_from_s3(s3_key, overwrite=False)



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s\t: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('segmenter app')
    logger.info('*** Starting Legend and Map Segmenter App ***')
    
    # prepare local data cache, and download model weights
    prepare_data_cache()

    # init segmenter
    segmenter = DetectronSegmenter(env.MODEL_CONFIG_YML, env.MODEL_WEIGHTS_PATH, confidence_thres=env.MODEL_CONFIDENCE_THRES)

    #### start flask server
    app.run(host='0.0.0.0', port=5000)

    #TEMP Use this for debug mode
    #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)