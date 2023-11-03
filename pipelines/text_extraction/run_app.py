from flask import Flask, request, Response
import logging, json
import numpy as np
#import cv2
#from urllib.parse import urlparse
import io
from PIL import Image

import env_defaults as env
from text_extractor import ResizeTextExtractor, TileTextExtractor, TextExtractor


#
# Flask web app for Legend and Map Segmenter module
#

#https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = 400000000      # to allow PIL to load large images

app = Flask(__name__)


@app.route('/api/process_image', methods=['POST'])
def process_image():
    '''
    Perform ocr text extraction on an image
    request.data is expected to contain binary image file buffer
    '''

    try:
        # TODO -- WIP, modify this POST request to receive JSON data? Including image bytes and additional run-time parameters 

        im = request.data
        # data = request.get_json()        
        # im = data.get('image', '')
        # do_tiling = data.get('do_tiling', False)
        # do_document_ocr = data.get('do_document_ocr', False)
        do_tiling = False
        do_document_ocr = False
        
        # if not im:
        #     msg = 'No "image" received. Need to provide "image" field in POST request containing binary image file buffer. Content-Type = application/json recommended.'
        #     logging.warning(msg)
        #     return (msg, 500)
        
        # decode and open as a PIL Image
        im = Image.open(io.BytesIO(im))

        #text_extractor: TextExtractor = None

        ocr_results: list = [] 
        if do_document_ocr:
            logger.info('Doing document-level OCR')
        if do_tiling:
            logger.info('Doing image tiling prior to OCR')
            # text-extraction with image tiling...
            ocr_results = tile_text_extractor.run(im, document_ocr=do_document_ocr)
        else:
            # OR, text-extraction with image resizing...
            ocr_results = resize_text_extractor.run(im, document_ocr=do_document_ocr)

        # convert result to a JSON array
        result_json = json.dumps([res.model_dump() for res in ocr_results])

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
    logger = logging.getLogger('text_extractor app')
    logger.info('*** Starting Text Extractor App ***')
    
    # init text extractors
    resize_text_extractor = ResizeTextExtractor(env.PIXEL_LIM)
    tile_text_extractor = TileTextExtractor(env.PIXEL_LIM)

    #### start flask server
    app.run(host='0.0.0.0', port=5000)

    #TEMP Use this for debug mode
    #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)