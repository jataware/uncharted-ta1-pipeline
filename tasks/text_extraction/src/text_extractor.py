
import numpy as np
import os, logging
from math import ceil
from PIL import Image

from ocr.google_vision_ocr import GoogleVisionOCR
from schema.page_extraction import PageExtraction
from schema.extraction_identifier import ExtractionIdentifier
from tile import Tile


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
        super().__init__()
        self.pixel_lim = pixel_lim
        self.model_id += f'-resize{pixel_lim}'


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


class TileTextExtractor(TextExtractor):
    '''
    OCR-based text extraction with image tiling prior to OCR
    '''
    
    def __init__(self, split_lim: int=PIXEL_LIM_DEFAULT):
        super().__init__()
        self.split_lim = split_lim
        self.model_id += f'-tile{split_lim}'
    

    def run(self, im: Image.Image, to_blocks: bool=True, document_ocr: bool=False) -> list[PageExtraction]:
        '''
        Run OCR-based text extractor
        Image may be internally tiled prior to OCR, if needed

        Args:
            im: input image (PIL image format)
            to_blocks: =True; group OCR results into blocks/lines of text related text
            document_ocr: =False; use 'document level' OCR, meant for images with dense paragraphs/columns of text
        Returns:
            List of PageExtraction objects
            (in pixel coords of full-sized image, not resized pixel coords)
        ''' 

        # TODO -- this code could be modified to include overlap/stride len, etc.
        # (then, any overlapping OCR results need to be de-dup'd)

        im_tiles = self._split_image(im, self.split_lim)
        logger.info(f'Image split into {len(im_tiles)} tiles. Extracting OCR text from each...')

        ocr_blocks = [] # list for OCR results across all tiles (whole image)
        for tile in im_tiles:
            # get OCR results for this tile
            tile_ocr_blocks = self._extract_text(tile.image, to_blocks, document_ocr)
            # convert OCR poly-bounds to global pixel coords and add to results
            ocr_blocks.extend(GoogleVisionOCR.offset_ocr_coords(tile_ocr_blocks, tile.coordinates))

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


    def _split_image(self, image: Image.Image, size_limit: int) -> list[Tile]:
        '''
        split an image as needed to fit under the image size limit for x and y
        '''

        image_size = image.size
        splits_x = self._get_splits(image_size[0], size_limit)
        splits_y = self._get_splits(image_size[1], size_limit)
        images = []
        for split_y in splits_y:
            for split_x in splits_x:
                ims = Image.new(mode="RGB", size=(split_x[1] - split_x[0], split_y[1] - split_y[0]))
                cropping = image.crop((split_x[0], split_y[0], split_x[1], split_y[1]))
                ims.paste(cropping, (0, 0))
                images.append(Tile(ims, (split_x[0], split_y[0])))
        return images


    def _get_splits(self, size: int, limit: int) -> list[tuple]:
        '''
        get the pixel intervals for image tiling 
        note, currently the tile stride == limit (0% overlap)
        '''
        splits = ceil(float(size) / limit)
        split_inc = ceil(float(size) / splits)
        split_vals = []
        current = 0
        while current < size:
            next_inc = min(current + split_inc, size)
            split_vals.append((current, next_inc))
            current = next_inc
        return split_vals
    
