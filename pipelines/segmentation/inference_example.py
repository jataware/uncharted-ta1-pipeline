import cv2
import os, glob
import logging
import env_defaults as env

from detectron_segmenter import DetectronSegmenter
import json

#
# Example script for processing an image
#

IMG_PATH_IN = 'path/to/input/image'     # EDIT THIS

logger = logging.getLogger(__name__)

def run():

    segmenter = DetectronSegmenter(env.MODEL_CONFIG_YML, env.MODEL_WEIGHTS_PATH, confidence_thres=env.MODEL_CONFIDENCE_THRES)

    logger.info('Processing image: {}'.format(IMG_PATH_IN))

    img = cv2.imread(IMG_PATH_IN)
    results = segmenter.run_inference(img, ta1_schema_out=True)

    # convert result to a JSON array
    result_json = json.dumps([res.model_dump() for res in results])
    print('RESULTS: ')
    print(result_json)

    logger.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s\t: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    run()