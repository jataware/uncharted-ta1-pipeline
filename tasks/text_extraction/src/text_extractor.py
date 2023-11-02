
import numpy as np
import os, logging
from math import ceil
from PIL import Image

from ocr.google_vision_ocr import GoogleVisionOCR
from schema.page_extraction import PageExtraction
from schema.extraction_identifier import ExtractionIdentifier


# ENV VARIABLE -- needed for google-vision API
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/path/to/google/vision/creds/json/file'

PIXEL_LIM_DEFAULT = 6000    # default max pixel limit for input image (determines amount of image resizing)

logger = logging.getLogger(__name__)


class TextExtractor():
    '''
    Base class for OCR-based text extraction
    '''

    def __init__(self):
        self.ocr = GoogleVisionOCR()
        self.model_id = 'google-cloud-vision'
    

    def _extract_text(self, im: Image.Image, to_blocks: bool=True, document_ocr: bool=False) -> list[dict[str, any]]:

        img_gv = GoogleVisionOCR.pil_to_vision_image(im)

        #----- do GoogleVision OCR
        ocr_texts = []
        if document_ocr:
            ocr_texts = self.ocr.detect_document_text(img_gv)
        else:
            ocr_texts = self.ocr.detect_text(img_gv)
            if to_blocks:
                ocr_texts = self.ocr.text_to_blocks(ocr_texts)

        return ocr_texts
    

    def run(self):
        # override in inherited classes below
        raise NotImplementedError
    

class ResizeTextExtractor(TextExtractor):
    '''
    OCR-based text extraction with optional image scaling prior to OCR
    '''

    def __init__(self, pixel_lim: int=PIXEL_LIM_DEFAULT):
        self.pixel_lim = pixel_lim
        super().__init__()

    
    def run(self, im: Image.Image, to_blocks: bool=True, document_ocr: bool=False) -> list[PageExtraction]:
        '''
        Run OCR-based text extractor
        Image may be internally scaled prior to OCR, if needed

        Args:
            im: input image (PIL image format)
            to_blocks: =True; group OCR results into blocks/lines of text related text
            document_ocr: =False; use 'document level' OCR, meant for images with dense paragraphs/columns of text
        Returns:
            List of PageExtraction objects
            (in pixel coords of full-sized image, not resized pixel coords)
        ''' 
        #im_orig_size = im.size   #(width, height)
        im_resized, im_resize_ratio = self._resize_image(im)

        ocr_blocks = self._extract_text(im_resized, to_blocks, document_ocr)

        # scale OCR pixel co-ords back to original image dimensions
        if ocr_blocks and im_resize_ratio < 1.0 and im_resize_ratio > 0.0:
            ocr_blocks = GoogleVisionOCR.scale_ocr_coords(ocr_blocks, 1.0/im_resize_ratio)

        # convert OCR results to TA1 schema
        # TODO -- could add in confidence per ocr block? (google vision seems to return score=0.0?)
        model_field_name = 'document_ocr' if document_ocr else 'ocr'

        results = []
        for ocr_block in ocr_blocks:
            ocr_result = PageExtraction(
                name='ocr',
                bounds=GoogleVisionOCR.bounding_polygon_to_list(ocr_block['bounding_poly']),
                ocr_text = ocr_block['text'],
                model=ExtractionIdentifier(model=self.model_id, field=model_field_name))
            results.append(ocr_result)

        return results
    

    def _resize_image(self, im: Image.Image) -> tuple[Image.Image, float]:
        '''
        Resize an image, if needed, so max dimension is <= self.pixel_lim
        '''
        # TODO could be moved to a 'common' module?

        im_orig_size = im.size   #(width, height)
        im_resize_ratio = 1.0
        if max(im_orig_size) > self.pixel_lim:
            im_resize_ratio = self.pixel_lim / max(im_orig_size)
            logger.info('Resizing image with ratio: {}'.format(im_resize_ratio))

            reduced_size = int(im_orig_size[0] * im_resize_ratio), int(im_orig_size[1] * im_resize_ratio)   
            im = im.resize(reduced_size, Image.Resampling.LANCZOS)
        
        return im, im_resize_ratio