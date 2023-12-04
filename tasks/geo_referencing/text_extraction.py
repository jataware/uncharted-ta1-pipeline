
import copy
import numpy as np

from math import ceil
from PIL.Image import Image as PILImage
from PIL import Image

from compute.ocr import OCR
from tasks.common.task import (Task, TaskInput, TaskResult)

from typing import (List, Optional)


PIXEL_LIM = 6000
JPG_TEMP_FILENAME = "temp/temp_image_resized.jpg"


ocr_cache = {}

class TextExtractor(Task):
    _ocr: OCR

    def __init__(self, task_id: str):
        super().__init__(task_id)
        self._ocr = OCR()
    
    def _get_ocr_text(self, image:PILImage, key:str=''):

        if key != '' and key in ocr_cache:
            print('reading from ocr cache')
            return copy.deepcopy(ocr_cache[key])
        
        # store image to temp location
        image.save(f'{JPG_TEMP_FILENAME}', "JPEG", quality=100, optimize=True)

        # ----- do GoogleVision OCR
        ocr_texts = self._ocr.detect_text(JPG_TEMP_FILENAME, key)
        #with open(f'temp/ocr-{key}.txt', "w") as f_out:
        #    f_out.write(f'{ocr_texts}')
        ocr_blocks = self._ocr.text_to_blocks(ocr_texts)
        #with open(f'temp/ocr-{key}-blocks.txt', "w") as f_out:
        #    f_out.write(f'{ocr_blocks}')
        if key != '':
            ocr_cache[key] = copy.deepcopy(ocr_blocks)
        
        return ocr_blocks


class ResizeTextExtractor(TextExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        print(f"running resizing text extraction task with id {self._task_id}")

        _, im_resize_ratio, image = self._resize_image(input.image)

        ocr_blocks = self._get_ocr_text(image, f'{input.raster_id}-resize')
        ocr_blocks = self._ocr.merge_multiline(ocr_blocks)
        #with open(f'temp/ocr-{input.raster_id}-resize-blocks-merged.txt', "w") as f_out:
        #    f_out.write(f'{ocr_blocks}')

        result = TaskResult()
        result.task_id = self._task_id
        result.output["ocr_blocks"] = ocr_blocks
        result.output["im_resize_ratio"] = im_resize_ratio
        return result

    def _resize_image(self, im: PILImage):
        im_orig_size = im.size  # (width, height)
        im_resize_ratio = 1
        if im.mode == "F":
            # floating point pixel format, need to convert to uint8
            np_im = np.asarray(im)
            pxl_mult = 255 / max(1.0, np.max(np_im))
            print(
                "WARNING! Raw image is in 32-bit floating point format. Scaling by {} and converting to uint8".format(
                    pxl_mult
                )
            )
            im = Image.fromarray(np.uint8(np_im * pxl_mult))
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")

        if max(im_orig_size) > PIXEL_LIM:
            im_resize_ratio = PIXEL_LIM / max(im_orig_size)
            print("Resizing image with ratio: {}".format(im_resize_ratio))

            reduced_size = int(im_orig_size[0] * im_resize_ratio), int(
                im_orig_size[1] * im_resize_ratio
            )
            im = im.resize(reduced_size, Image.Resampling.LANCZOS)

        return im_orig_size, im_resize_ratio, im


class TileTextExtractor(TextExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        print(f"running tiling text extraction task with id {self._task_id}")

        ims = self._split_image(input.image, PIXEL_LIM)

        ocr_blocks = []
        for im in ims:
            ocr = self._get_ocr_text(im.image, f'{input.raster_id}-tile-{im.coordinates[0]}-{im.coordinates[1]}')
            ocr_blocks = ocr_blocks + self._adjust_ocr_blocks(ocr, im.coordinates)
        #with open(f'temp/ocr-{input.raster_id}-blocks.txt', "w") as f_out:
        #    f_out.write(f'{ocr_blocks}')
        ocr_blocks = self._ocr.merge_multiline(ocr_blocks)
        #with open(f'temp/ocr-{input.raster_id}-blocks-merged.txt', "w") as f_out:
        #    f_out.write(f'{ocr_blocks}')

        result = TaskResult()
        result.task_id = self._task_id
        result.output["ocr_blocks"] = ocr_blocks
        return result

    def _split_image(self, image: PILImage, size_limit: int):
        # split an image as needed to fit under the image size limit for x and y
        image_size = image.size
        splits_x = self._get_splits(image_size[0], size_limit)
        splits_y = self._get_splits(image_size[1], size_limit)
        images = []
        for split_y in splits_y:
            for split_x in splits_x:
                ims = Image.new(
                    mode="RGB", size=(split_x[1] - split_x[0], split_y[1] - split_y[0])
                )
                cropping = image.crop((split_x[0], split_y[0], split_x[1], split_y[1]))
                ims.paste(cropping, (0, 0))
                images.append(Tile(ims, (split_x[0], split_y[0])))
        return images

    def _get_splits(self, size: int, limit: int) -> List[tuple[int, int]]:
        splits = ceil(float(size) / limit)
        split_inc = ceil(float(size) / splits)
        split_vals = []
        current = 0
        while current < size:
            next_inc = min(current + split_inc, size)
            split_vals.append((current, next_inc))
            current = next_inc
        return split_vals

    def _adjust_ocr_blocks(self, ocr_blocks, offset: tuple[int, int]):
        # offset all x & y by the (x,y) coordinates
        # does the adjustment in-place
        for b in ocr_blocks:
            for v in b["bounding_poly"].vertices:
                v.x = v.x + offset[0]
                v.y = v.y + offset[1]

        return ocr_blocks


class Tile:
    image: Optional[PILImage] = None
    coordinates: tuple[int, int] = (0, 0)

    def __init__(self, image: PILImage, coordinates: tuple[int, int]):
        self.image = image
        self.coordinates = coordinates
