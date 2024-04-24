import cv2
import re
import logging
import numpy as np

from tasks.text_extraction.entities import TextExtraction
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from typing import List, Tuple

# TODO TEMP check these
COLOUR_RANGE_L = 40  # for separating foreground vs background pixels (lower == more aggressive foregnd masking)
COLOUR_RANGE_AB = 35
WHITE_SHIFT = 100  # emphasize the foreground features by whitening
WHITE_LAB = (255, 128, 128)  # white in LAB colour-space

# matches chars that are not numbers or letters
RE_NON_ALPHANUMERIC = re.compile(r"[^0-9A-Za-z]")

logger = logging.getLogger(__name__)


def build_ocr_index(text_extractions: List[TextExtraction]) -> STRtree:
    """
    build a shapely STR-tree index for OCR text blocks
    """
    ocr_poly_list = []
    for text_extr in text_extractions:
        bounds = [(pt.x, pt.y) for pt in text_extr.bounds]
        p = Polygon(bounds)
        ocr_poly_list.append(p)

    return STRtree(ocr_poly_list)


def get_colour_range(median_values: List, colour_ranges: List) -> Tuple:
    """
    get upper and lower colour range values, centered around target median colour values
    for image colour-based filtering

    """
    colour_lower = []
    colour_upper = []

    for med_val, c_range in zip(median_values, colour_ranges):
        colour_lower.append(max(med_val - c_range, 0))
        colour_upper.append(min(med_val + c_range, 255))

    return np.array(colour_lower, dtype=np.uint8), np.array(
        colour_upper, dtype=np.uint8
    )


def crop_template(
    template: np.ndarray,
    fore_mask: np.ndarray,
    crop_buffer=5,
    backgnd_colour=(255, 255, 255),
) -> np.ndarray:
    """
    Crop template image based on foreground mask
    and add background buffer pixels along the border
    """

    fy, fx = np.where(fore_mask != 0)
    template = cv2.copyMakeBorder(
        template,
        crop_buffer,
        crop_buffer,
        crop_buffer,
        crop_buffer,
        cv2.BORDER_CONSTANT,
        value=backgnd_colour,
    )
    fy = fy + crop_buffer
    fx = fx + crop_buffer
    h, w = template.shape[0], template.shape[1]
    tx_min = max(fx.min() - crop_buffer, 0)
    tx_max = min(fx.max() + crop_buffer + 1, w - 1)
    ty_min = max(fy.min() - crop_buffer, 0)
    ty_max = min(fy.max() + crop_buffer + 1, h - 1)
    template = template[ty_min:ty_max, tx_min:tx_max]

    return template


def template_pre_processing(
    im: np.ndarray, im_templ: np.ndarray, im_templ_raw: np.ndarray
) -> Tuple:
    """
    pre-process the main image and template image, prior to template matching
    (assume input images are opencv format with RGB colour format)

    Returns the pre-processed image and template image in RGB colour space
    """

    # ---- Foreground and background colour analysis of the template image
    # Perform Otsu thresholding and extract the foreground
    templ_thres, fore_mask = cv2.threshold(
        cv2.cvtColor(im_templ, cv2.COLOR_RGB2GRAY),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )  # Currently foreground is only a mask

    if im_templ_raw.size > 0:
        fore_y, fore_x = np.where(fore_mask > 0)  # type: ignore
        if len(fore_x) < im_templ.shape[0] * im_templ.shape[1] * 0.01:
            # less than 1 percent of template image is 'foreground'?
            # oops! try without masking OCR!
            logger.warning("OCR masking too aggressive for template image")
            im_templ = im_templ_raw
            templ_thres, fore_mask = cv2.threshold(
                cv2.cvtColor(im_templ, cv2.COLOR_RGB2GRAY),
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

    # ---- Convert to LAB colour-space
    im_templ = cv2.cvtColor(im_templ, cv2.COLOR_RGB2LAB)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    # get the 'foreground' pixel values from the template, and
    # get template colour stats
    idx = fore_mask != 0
    templ_fore = im_templ[idx]
    colour_med_val = np.median(templ_fore, axis=0)  # type: ignore

    colour_lower, colour_upper = get_colour_range(
        colour_med_val.tolist(), [COLOUR_RANGE_L, COLOUR_RANGE_AB, COLOUR_RANGE_AB]
    )

    # ---- De-emphasize (whiten) non-foreground pixels in the main image
    # create a mask to separate foreground and background pixels
    im_mask = cv2.inRange(im, colour_lower, colour_upper)

    kernel_morph = np.ones((3, 3), np.uint8)
    im_mask = cv2.dilate(im_mask, kernel_morph, iterations=1)
    im_mask = cv2.bitwise_not(im_mask)  # mask now for background pixels

    # emphasize the foreground features by whitening the 'background' pixels
    # (perform whitening in RGB colour-space)
    im = cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    im = cv2.add(im, WHITE_SHIFT, mask=im_mask)  # type: ignore
    im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    # final cropping of the template and size normalization
    im_templ = crop_template(
        im_templ, fore_mask, crop_buffer=5, backgnd_colour=WHITE_LAB
    )

    # convert results back to RGB colour space
    im_templ = cv2.cvtColor(im_templ, cv2.COLOR_LAB2RGB)
    im = cv2.cvtColor(im, cv2.COLOR_LAB2RGB)

    return (im, im_templ)


def template_matching(im: np.ndarray, im_templ: np.ndarray, search_range=-1) -> Tuple:
    """
    perform opencv template matching between base and template images

    search_range: restrict xcorr to pixels around the center of the base image

    returns max x-correlation value and x,y pixel location
    """

    # ---- find the template match values for a given template
    im_xcorr = cv2.matchTemplate(im, im_templ, cv2.TM_CCOEFF_NORMED)

    # note: im_xcorr has dimensions (W-w+1, H-h+1)
    # so each im_xcorr pxel maps to the orig image pixel - half the template size (in each dimension)
    # xcorr x = base image x - (template width / 2)
    # xcorr y = base image y - (template height / 2)
    # https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

    im_xcorr = np.nan_to_num(im_xcorr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    y_offset = int(im_templ.shape[0] / 2)
    x_offset = int(im_templ.shape[1] / 2)

    # at im base center
    y_c = int(im.shape[0] / 2)
    x_c = int(im.shape[1] / 2)

    if search_range >= 0:
        # restrict xcorr to search_range pixels around the center of the base image
        search_half = int(search_range / 2)
        ymin = max(y_c - y_offset - search_half, 0)
        xmin = max(x_c - x_offset - search_half, 0)
        ymax = min(y_c - y_offset + search_half, im_xcorr.shape[0])
        xmax = min(x_c - x_offset + search_half, im_xcorr.shape[1])

        im_xcorr = im_xcorr[ymin:ymax, xmin:xmax]

        x_offset -= xmin
        y_offset -= ymin

    # https://stackoverflow.com/questions/55284090/how-to-find-maximum-value-in-whole-2d-array-with-indices
    max_xy = np.unravel_index(im_xcorr.argmax(), im_xcorr.shape)
    max_val = im_xcorr[max_xy]

    # convert max indices back to 'base image' pixel locations...
    max_xy = (int(max_xy[0]) + y_offset, int(max_xy[1]) + x_offset)

    return max_val, max_xy


def angle_in_range(theta: float, a: float, b: float) -> bool:
    """
    check if angle theta is in the range of angle a to angle b, inclusive (CCW dir'n)
    - all angles assumed to be in units of degrees
    - all angles assumed to be positive values
    """
    # adapted from https://math.stackexchange.com/questions/3111484/how-to-determine-if-a-value-falls-within-a-specific-angle-range-on-a-circle
    theta1 = theta % 360  # ensure all angles are 0 to 360 deg range
    a1 = a % 360
    b1 = b % 360
    if a1 <= b1:
        # angle range doesn't wrap around 0 deg
        return theta1 >= a1 and theta1 <= b1
    else:
        # angle does wrap around 0 deg
        return theta1 >= a1 or theta1 <= b1


def mask_ocr_blocks(
    im: np.ndarray,
    text_extractions: List[TextExtraction],
    max_area: int,
    ybuffer: int = 0,
    min_len: int = 1,
    prune_symbols=False,
):
    """
    mask text blocks with median pixel values
    """

    if text_extractions:
        for blk in text_extractions:
            prose = blk.text.strip()
            if prune_symbols:
                prose = RE_NON_ALPHANUMERIC.sub("", prose).strip()
            if min_len > 0 and len(prose) <= min_len:
                continue

            p = Polygon([(pt.x, pt.y) for pt in blk.bounds])
            (xmin, ymin, xmax, ymax) = [int(b) for b in p.bounds]
            if xmin == xmax or ymin == ymax:
                continue
            ocr_blk_area = (ymax - ymin) * (xmax - xmin)
            if max_area > 0 and ocr_blk_area > max_area:
                # area is too big, skip masking
                continue
            if ybuffer > 0:
                # ymin = max(ymin-ybuffer,0)
                ymax += ybuffer
            ocr_pxl_slice = im[
                ymin:ymax, xmin:xmax, :
            ]  # TODO - or just get median val along the top?
            if ocr_pxl_slice.size > 0:
                med_val = np.median(ocr_pxl_slice, axis=[0, 1])
                ocr_pxl_slice[:, :, 0] = med_val[0]
                ocr_pxl_slice[:, :, 1] = med_val[1]
                ocr_pxl_slice[:, :, 2] = med_val[2]

    return im
