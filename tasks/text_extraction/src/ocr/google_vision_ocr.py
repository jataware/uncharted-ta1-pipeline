
from google.cloud.vision import ImageAnnotatorClient
from google.cloud.vision import Image as VisionImage
from google.cloud.vision_v1.types.geometry import BoundingPoly

from PIL import Image
import cv2
import numpy as np
import io, copy, logging

logger = logging.getLogger(__name__)


class GoogleVisionOCR():
    '''
    Extract text from an image using Google Vision's OCR API
    '''

    def detect_text(self, vision_img: VisionImage) -> list[dict[str, any]]:
        '''
        Runs google vision OCR on an image and returns the text annotations as a dictionary

        Args:
            vision_img: input image (google vision Image class)

        Returns:
            List of text extractions -- {"text": <extracted text>, "bounding_poly": <google BoundingPoly object>, "confidence": <extraction confidence>}
        '''
        # (from https://cloud.google.com/vision/docs/ocr#vision_text_detection-python)
        # TODO -- Note: extraction 'score' always seems to be 0.0 ??
        
        client = ImageAnnotatorClient()
        response = client.text_detection(image=vision_img)  # type: ignore

        text_extractions = []
        if response.text_annotations:
            #note: first entry will be the entire text block
            text_extractions = [
                {"text": text.description, "bounding_poly": text.bounding_poly, "confidence": text.score}   # text.confidence is deprecated
                for text in response.text_annotations
            ]
        else:
            logger.warning('No OCR text found!')

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return text_extractions
    

    def text_to_blocks(self, texts: list[dict[str, any]]) -> list[dict[str, any]]:
        '''
        Clean up extract OCR text list into blocks of continuus text (lines)
        and adjust bounding-boxes (polygons) as needed
        NOTE:
        - meant to be used in conjunction with `detect_text` func
        - assumes first element contains the full text delimited by line breaks)
        '''

        if len(texts) < 2:
            logger.warning('Less than 2 blocks of OCR text found! Skipping ocr_text_to_blocks')
            return []

        full_text = texts[0]['text']
    
        group_offset = 1
        num_blocks = 0
        group_counter = 0

        results: list[dict[str, any]] = []
        
        text_blocks = full_text.split('\n')
        text_block = text_blocks[0]
        for text_block_next in text_blocks[1:]:
            
            text_block = text_block.strip()
            text_block0 = text_block
            bounding_poly = None #vision.BoundingPoly()
            for text in texts[group_offset:]:
                prose = text['text'].strip()
                group_counter += 1
                text_block_sub = text_block.replace(prose, '', 1).strip()       # TODO could make this replace from the start of string...
                if len(text_block_sub) == 0:
                    bounding_poly = self._add_bounding_polygons(bounding_poly, text['bounding_poly'])
                    break
                elif len(prose) > 0 and len(text_block_sub) == len(text_block) and text_block_next.startswith(prose):   # and len(text_block_sub) < 3
                    # partial OCR block parsed! ... finish with this block and go to next one
                    group_counter -= 1      
                    break
                bounding_poly = self._add_bounding_polygons(bounding_poly, text['bounding_poly'])
                text_block = text_block_sub

            num_blocks += 1

            # TODO could try this too.. for bounding_poly
            #from google.protobuf.json_format import MessageToDict
            #dict_obj = MessageToDict(org)
            if bounding_poly is not None:
                results.append({'text' : text_block0, 'bounding_poly' : bounding_poly})
            group_offset += group_counter
            group_counter = 0
            text_block = text_block_next

        # and save last block too
        # results.append({'text' : text_block.strip(), 'bounding_poly' : bounding_poly})
        if len(results) < len(text_blocks) - 1:
            logger.warning('Possible error grouping OCR results')   # TODO - throw exception here?
        
        return results
    

    def detect_document_text(self, vision_img: VisionImage) -> list[dict[str, any]]:
        '''
        Runs google vision OCR on the image and returns the text annotations as a dictionary
        '''

        client = ImageAnnotatorClient()
        response = client.document_text_detection(image=vision_img)  # type: ignore

        text_extractions = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_words = []
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([symbol.text for symbol in word.symbols])
                            block_words.append(word_text)

                    text_extractions.append(
                        {
                            "text": " ".join(block_words),
                            "bounding_poly": block.bounding_box,
                            "confidence": block.confidence
                        }
                    )
        else:
            logger.warning("No OCR text found!")

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )
        
        return text_extractions


    def _add_bounding_polygons(self, poly1: BoundingPoly, poly2: BoundingPoly) -> BoundingPoly:
        '''
        Adds two google vision 'bounding polygon' objects together to create a new bounding polygon that contains both
        NOTE: this returns a bounding box, as opposed to a arbitrary-shaped polygon
        '''

        # TODO -- could return a polygon of merged bounding polygons
        # e.g., using Shapely 'union' funcs

        if poly1 is None:
            return poly2
        if poly2 is None:
            return poly1

        min_x = poly1.vertices[0].x
        min_y = poly1.vertices[0].y
        max_x = poly1.vertices[0].x
        max_y = poly1.vertices[0].y

        for vertex in poly1.vertices:
            min_x  = min(min_x, vertex.x)
            min_y  = min(min_y, vertex.y)
            max_x  = max(max_x, vertex.x)
            max_y  = max(max_y, vertex.y)

        for vertex in poly2.vertices:
            min_x  = min(min_x, vertex.x)
            min_y  = min(min_y, vertex.y)
            max_x  = max(max_x, vertex.x)
            max_y  = max(max_y, vertex.y)

        poly_combined = copy.deepcopy(poly1)
        poly_combined.vertices[0].x = min_x
        poly_combined.vertices[1].x = max_x
        poly_combined.vertices[2].x = max_x
        poly_combined.vertices[3].x = min_x

        poly_combined.vertices[0].y = min_y
        poly_combined.vertices[1].y = min_y
        poly_combined.vertices[2].y = max_y
        poly_combined.vertices[3].y = max_y

        return poly_combined
    

    @staticmethod
    def scale_ocr_coords(text_blocks: list[dict[str, any]], resize_factor: float) -> list[dict[str, any]]:
        '''
        scale ocr pixel coords by a re-size factor 
        '''

        if not text_blocks or resize_factor == 1.0:
            return text_blocks
        # TODO could add a check that any bounding_poly coords are within original image boundaries
        for blk in text_blocks:
            for v in blk['bounding_poly'].vertices:
                v.x = int(v.x * resize_factor)
                v.y = int(v.y * resize_factor)  

        return text_blocks
    

    @staticmethod
    def offset_ocr_coords(text_blocks: list[dict[str, any]], offset: tuple) -> list[dict[str, any]]:
        '''
        offset ocr pixel coords by (x,y) co-ords 
        '''

        if not text_blocks or not offset:
            return text_blocks
        for blk in text_blocks:
            for v in blk['bounding_poly'].vertices:
                v.x = int(v.x + offset[0])
                v.y = int(v.y + offset[0])  

        return text_blocks
    

    @staticmethod
    def bounding_polygon_to_list(bounding_poly: BoundingPoly) -> list[tuple]:
        '''
        convert google vision BoundingPoly object to list of x,y points
        '''
        poly_pts = [
            (vertex.x, vertex.y)
            for vertex in bounding_poly.vertices
        ]

        return poly_pts
    

    # TODO -- some of these static methods could go into a 'common' module??
    # E.g., esp the load_* image functions?

    @staticmethod
    def load_vision_image(path: str) -> VisionImage:
        '''
        Loads an image into memory as a google vision api image object
        '''
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        return VisionImage(content=content)

    @staticmethod
    def pil_to_vision_image(pil_image: Image.Image) -> VisionImage:
        '''
        Converts a PIL image object to a google vision api image object
        '''
        pil_image_bytes = io.BytesIO()
        pil_image.save(pil_image_bytes, format="JPEG")
        return VisionImage(content=pil_image_bytes.getvalue())
    
    @staticmethod
    def cv_to_vision_image(cv_image: np.ndarray) -> VisionImage:
        '''
        Converts an opencv image (numpy array) to a google vision api image object
        '''
        # (from https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/)
        success, encoded_image = cv2.imencode('.jpg', cv_image)
        return VisionImage(content=encoded_image.tobytes())
