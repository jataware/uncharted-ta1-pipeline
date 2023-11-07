
import os
import logging, json

from ocr.google_vision_ocr import GoogleVisionOCR
from text_extractor import ResizeTextExtractor, TileTextExtractor
from image_io import load_pil_image, normalize_image_format

# ENV VARIABLE -- needed for google-vision API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/path/to/google/vision/creds/json/file'

def run():

    # Example code to load an image and perform text extraction
    image_path = 'path/to/input/image'
    logging.info('')
    logging.info(f'Running text extraction on image {image_path}')
    
    im = load_pil_image(image_path)
    im = normalize_image_format(im)

    # text-extraction with image scaling...
    text_extractor = ResizeTextExtractor() 
    # OR, text-extraction with image tiling...
    #text_extractor = TileTextExtractor()

    ocr_results = text_extractor.run(im)

    # convert result to a JSON array
    results_json = json.dumps([res.model_dump() for res in ocr_results])
    print(results_json)

    logging.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s\t: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    run()

