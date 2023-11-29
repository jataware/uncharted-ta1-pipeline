#
# Segmentation Pipeline -- Environment values and constants
#

# TODO move some of this to a common dir?

import os
from dotenv import load_dotenv


def set_param(param_name, default_val=None, verbose=True):
    param_value = (
        os.environ.get(param_name)
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

# Model config
MODEL_DATA_PATH = set_param("MODEL_DATA_PATH")
MODEL_CONFIDENCE_THRES = float(set_param("MODEL_CONFIDENCE_THRES", "0.25"))  # type: ignore

# Data caching
# (ie for storing downloaded model in a docker volume)
MODEL_DATA_CACHE_PATH = set_param("MODEL_DATA_CACHE_PATH")

# S3 params
S3_HOST = set_param("S3_HOST", "https://s3.t1.uncharted.software")
AWS_ACCESS_KEY_ID = set_param(
    "AWS_ACCESS_KEY_ID", default_val=None, verbose=False
)  # don't print s3 credentials
AWS_SECRET_ACCESS_KEY = set_param(
    "AWS_SECRET_ACCESS_KEY", default_val=None, verbose=False
)

# Output results in TA1 schema?
USE_TA1_SCHEMA = set_param("USE_TA1_SCHEMA", "true").lower() == "true"  # type: ignore
