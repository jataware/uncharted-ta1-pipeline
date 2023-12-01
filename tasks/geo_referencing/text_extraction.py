import numpy as np

from math import ceil
from PIL import Image

from compute.ocr import OCR
from tasks.geo_referencing.task import Task, TaskInput, TaskResult


PIXEL_LIM = 6000
JPG_TEMP_FILENAME = "temp/temp_image_resized.jpg"


class TextExtractor(Task):
    _ocr: OCR = None

    def __init__(self, task_id: str):
        super().__init__(task_id)
        self._ocr = OCR()

    def _get_ocr_text(self, image: Image, key: str = ""):
        # store image to temp location
        image.save(f"{JPG_TEMP_FILENAME}", "JPEG", quality=100, optimize=True)

        # ----- do GoogleVision OCR
        ocr_texts = self._ocr.detect_text(JPG_TEMP_FILENAME, key)
        ocr_blocks = self._ocr.text_to_blocks(ocr_texts)

        return ocr_blocks


class ResizeTextExtractor(TextExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        print(f"running resizing text extraction task with id {self._task_id}")

        _, im_resize_ratio, image = self._resize_image(input.image)

        ocr_blocks = self._get_ocr_text(image, f"{input.raster_id}-resize")

        result = TaskResult()
        result.task_id = self._task_id
        result.output["ocr_blocks"] = ocr_blocks
        result.output["im_resize_ratio"] = im_resize_ratio
        return result

    def _resize_image(self, im: Image):
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
            ocr = self._get_ocr_text(
                im.image, f"{input.raster_id}-tile-{im.coordinates}"
            )
            ocr_blocks = ocr_blocks + self._adjust_ocr_blocks(ocr, im.coordinates)

        result = TaskResult()
        result.task_id = self._task_id
        result.output["ocr_blocks"] = ocr_blocks
        return result

    def _split_image(self, image: Image, size_limit: int):
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

    def _get_splits(self, size: int, limit: int) -> list((int, int)):
        splits = ceil(float(size) / limit)
        split_inc = ceil(float(size) / splits)
        split_vals = []
        current = 0
        while current < size:
            next_inc = min(current + split_inc, size)
            split_vals.append((current, next_inc))
            current = next_inc
        return split_vals

    def _adjust_ocr_blocks(self, ocr_blocks, offset: (int, int)):
        # offset all x & y by the (x,y) coordinates
        # does the adjustment in-place
        for b in ocr_blocks:
            for v in b["bounding_poly"].vertices:
                v.x = v.x + offset[0]
                v.y = v.y + offset[1]

        return ocr_blocks


class Tile:
    image: Image = None
    coordinates: (int, int) = (0, 0)

    def __init__(self, image: Image, coordinates: (int, int)):
        self.image = image
        self.coordinates = coordinates
