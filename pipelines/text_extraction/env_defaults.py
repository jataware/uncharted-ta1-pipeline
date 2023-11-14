#
# Text Extraction Pipeline -- Environment values and constants
#

# TODO move some of this to a common dir?

import os
from dotenv import load_dotenv
from typing import Optional


def set_param(param_name: str, default_val: str = "", verbose=True) -> str:
    param_value = (
        os.environ.get(param_name, default_val)
        if default_val is None
        else os.environ.get(param_name, default_val)
    )
    if verbose:
        print("{}\t:\t{}".format(param_name, param_value))
    return param_value


if not os.environ.get("ENV_LOADED"):
    # load environment variables from .env file
    print("\nLoading environment variables from .env")
    load_dotenv(verbose=True)
    if not os.environ.get("ENV_LOADED"):
        # .env still not loaded properly!
        raise Exception("Unable to load environment variables from .env")


# ---- Set environment variables...
print("\n*** Setting environment variables:")

# google vision credentials json file
GOOGLE_APPLICATION_CREDENTIALS = set_param("GOOGLE_APPLICATION_CREDENTIALS")

# default max pixel limit for input image
# (determines amount of image resizing or tiling prior to OCR)
PIXEL_LIM = int(set_param("PIXEL_LIM", "6000"))
