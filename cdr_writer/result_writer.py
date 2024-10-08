import argparse
import logging
import os
from cdr_writer.write_result_subscriber import WriteResultSubscriber
from tasks.common.request_client import WRITE_REQUEST_QUEUE
from util.logging import config_logger

logger = logging.getLogger("cdr")

CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]
CDR_HOST = "https://api.cdr.land"

REQUEUE_LIMIT = 3
INACTIVITY_TIMEOUT = 5
HEARTBEAT_INTERVAL = 900
BLOCKED_CONNECTION_TIMEOUT = 600


def main():
    # default log settings
    config_logger(logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--imagedir", type=str, required=True)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--cdr_host", type=str, default=CDR_HOST)
    p = parser.parse_args()

    result_subscriber = WriteResultSubscriber(
        WRITE_REQUEST_QUEUE,
        p.cdr_host,
        CDR_API_TOKEN,
        p.output,
        p.workdir,
        p.imagedir,
        host=p.host,
        port=p.rabbit_port,
    )
    result_subscriber.start_lara_result_queue()


if __name__ == "__main__":
    main()
