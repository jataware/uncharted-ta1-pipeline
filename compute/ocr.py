
from google.cloud import vision
import io, copy
import os
import pickle


class OCR:

    def __init__(self):
        pass

    #
    # OCR of an image using Google Vision API
    # https://cloud.google.com/vision/docs/ocr#vision_text_detection-python
    #
    def detect_text(self, path, key:str=''):
        """Detects text in the file."""
        
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)

        texts = []
        if response.text_annotations:
            texts = response.text_annotations
        else:
            print('WARNING! No OCR text found!')

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return texts

    #
    # text_to_blocks
    #
    # clean up extract OCR text list into blocks of continuus text (lines)
    # and adjust bounding-boxes (polygons) as needed
    # (assumed first element contains the full text delimited by line breaks)
    #
    def text_to_blocks(self, texts):

        if len(texts) < 2:
            print('WARNING! less than 2 blocks of OCR text found! Skipping ocr_text_to_blocks')
            return []

        full_text = texts[0].description
    
        group_offset = 1
        num_blocks = 0
        group_counter = 0
        results = []
        text_blocks = full_text.split('\n')
        text_block = text_blocks[0]
        for text_block_next in text_blocks[1:]:
            
            text_block = text_block.strip()
            text_block0 = text_block
            bounding_poly = None #vision.BoundingPoly()
            for text in texts[group_offset:]:
                prose = text.description.strip()
                group_counter += 1
                text_block_sub = text_block.replace(prose, '', 1).strip()       # TODO could make this replace from the start of string...
                if len(text_block_sub) == 0:
                    bounding_poly = self._add_bounding_polygons(bounding_poly, text.bounding_poly)
                    break
                elif len(prose) > 0 and len(text_block_sub) == len(text_block) and text_block_next.startswith(prose):   # and len(text_block_sub) < 3
                    # partial OCR block parsed! ... finish with this block and go to next one
                    group_counter -= 1      
                    break
                bounding_poly = self._add_bounding_polygons(bounding_poly, text.bounding_poly)
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
            print('WARNING! Possible error grouping OCR results')   # TODO - throw exception here?
        
        return results
    
    def merge_multiline(self, ocr_blocks):
        print('looking for multiline ocr blocks')
        # process blocks top to bottom
        blocks_sorted = sorted(ocr_blocks, key = lambda x: x['bounding_poly'].vertices[0].y)

        # merge blocks that are closely aligned horizontally, offset vertically by roughly one line height
        blocks_combined = []
        merged = {}
        x_match_limit = 15
        for i, b in enumerate(blocks_sorted):
            # make sure the ocr block is not already combined
            vertices = b['bounding_poly'].vertices
            if (vertices[0].x, vertices[0].y) in merged:
                continue

            # check for blocks that are vertically near
            # near defined as having their top corners within one matching bounding box size
            lower_bound = vertices[2].y
            upper_bound = lower_bound + vertices[2].y - vertices[0].y

            # for now limit to one join rather than multiple joins
            # data is sorted on y so only entries further in the list can be matches
            ocr_match = None
            for b2 in blocks_sorted[i + 1:]:
                vs = b2['bounding_poly'].vertices
                # check for exceeding y distance limit for merging
                if vs[0].y > upper_bound:
                    break
                
                # confirm y falls within expected bound
                # given the data is sorted, this should not be needed but meh
                if vs[0].y < lower_bound:
                    continue

                # x boundary needs to fall within an arbitrary pixel limit (in this case +- 15) on both sides
                if abs(vs[0].x - vertices[0].x) < x_match_limit and abs(vs[2].x - vertices[2].x) < x_match_limit:
                    ocr_match = b2
                    break
            c = b
            if ocr_match:
                #print('FOUND MULTILINE TO MERGE')
                #print((ocr_match['bounding_poly'].vertices[0].x, ocr_match['bounding_poly'].vertices[0].y))
                #print(f'TEXT:\n{b["text"]}\nAND\n{ocr_match["text"]}')
                combined_bb = self._add_bounding_polygons(b['bounding_poly'], ocr_match['bounding_poly'])
                c = {'text' : f'{b["text"]} {ocr_match["text"]}', 'bounding_poly' : combined_bb}
                merged[(ocr_match['bounding_poly'].vertices[0].x, ocr_match['bounding_poly'].vertices[0].y)] = ocr_match
            blocks_combined.append(c)
            

        return blocks_combined

    #
    # add_bounding_polygons
    #
    def _add_bounding_polygons(self, poly1, poly2):

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