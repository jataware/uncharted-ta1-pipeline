#
# Code for loading/setting global environment variables
#
import logging, os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DOTENV_PATH_DEFAULT = '.env'


def set_param(param_name, default_val=None, verbose=True):

    param_value = os.environ.get(param_name) if default_val is None else os.environ.get(param_name, default_val)
    if verbose:
        print('{}\t:\t{}'.format(param_name, param_value))
    return param_value


if not os.environ.get('SEGMENTER_ENV_LOADED'):
    # load environment variables from .env file
    print('\nLoading environment variables from ' + DOTENV_PATH_DEFAULT)
    load_dotenv(dotenv_path=DOTENV_PATH_DEFAULT, verbose=True)
    if not os.environ.get('SEGMENTER_ENV_LOADED'):
        # .env still not loaded properly!
        raise Exception('Unable to load environment variables from ' + DOTENV_PATH_DEFAULT)


# set environment variables...
print('\n*** Setting environment variables:')

SEGMENTER_DATA_CACHE = set_param('SEGMENTER_DATA_CACHE')
MODEL_WEIGHTS = set_param('MODEL_WEIGHTS')
MODEL_CONFIG_YML = set_param('MODEL_CONFIG_YML')
MODEL_CONFIDENCE_THRES = float(set_param('MODEL_CONFIDENCE_THRES'))
