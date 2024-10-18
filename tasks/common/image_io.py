from PIL import Image
from PIL.Image import Image as PILImage
import cv2
import numpy as np
import logging
from io import BytesIO

#
# Generic image loading/saving and formatting functions
#

# https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
Image.MAX_IMAGE_PIXELS = 400000000  # to allow PIL to load large images

logger = logging.getLogger(__name__)


def load_pil_image(path: str, normalize=True) -> PILImage:
    """
    Loads an image into memory as a PIL image object
    """
    image = Image.open(path)
    if normalize:
        image = normalize_image_format(image)
    return image


def load_pil_image_stream(bytes_io: BytesIO, normalize=True) -> PILImage:
    """
    Loads an image into memory as a PIL image object from a byte-stream
    """
    image = Image.open(bytes_io)
    if normalize:
        image = normalize_image_format(image)
    return image


def load_cv_image(path: str) -> np.ndarray:
    """
    Loads an image into memory as a opencv image object (numpy array)
    """
    return cv2.imread(path)


def pil_to_cv_image(pil_image: PILImage, rgb2bgr: bool = False) -> np.ndarray:
    """
    Converts a PIL image object to an opencv image (np array)
    NOTE: Pillow images are usually RGB format by default, whereas OpenCV are usually BGR
    """
    if rgb2bgr:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        return np.array(pil_image)


def cv_to_pil_image(cv_image: np.ndarray, bgr2rgb: bool = False) -> PILImage:
    """
    Converts an opencv image object (np array) to a Pillow image
    NOTE: Pillow images are usually RGB format by default, whereas OpenCV are usually BGR
    """
    if bgr2rgb:
        return Image.fromarray(cv2.cvtColor(np.array(cv_image), cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray(cv_image)


def normalize_image_format(im: PILImage) -> PILImage:
    """
    Normalize a Pillow image to RGB color format
    with uint8 pixel values
    """
    if im.mode == "F":
        # floating point pixel format, need to convert to uint8
        np_im = np.asarray(im)
        pxl_mult = 255 / max(1.0, np.max(np_im))
        logger.warning(
            "Raw image is in 32-bit floating point format. Scaling by {} and converting to uint8".format(
                pxl_mult
            )
        )
        im = Image.fromarray(np.uint8(np_im * pxl_mult))
    if im.mode in ("RGBA", "P", "1", "L"):
        im = im.convert("RGB")
    return im
