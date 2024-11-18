import argparse
import atexit
import httpx
import json
import logging
import ngrok
import os
import requests

from flask import Flask, request, Response

from cdr.chaining_result_subscriber import ChainingResultSubscriber
from cdr.request_publisher import LaraRequestPublisher
from tasks.common.request_client import (
    GEO_REFERENCE_REQUEST_QUEUE,
    METADATA_REQUEST_QUEUE,
    POINTS_REQUEST_QUEUE,
    SEGMENTATION_REQUEST_QUEUE,
    Request,
)

from schema.mappers.cdr import get_mapper
from schema.cdr_schemas.events import MapEventPayload

from typing import Any, Dict, List, Optional

from util.logging import config_logger

logger = logging.getLogger("cdr")

app = Flask(__name__)

CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]
CDR_HOST = "https://api.cdr.land"
COG_PATH = "https://s3.amazonaws.com/public.cdr.land/cogs"
CDR_CALLBACK_SECRET = "maps rock"
APP_PORT = 5001

LARA_RESULT_QUEUE_NAME = "lara_result_queue"

REQUEUE_LIMIT = 3
INACTIVITY_TIMEOUT = 5
HEARTBEAT_INTERVAL = 900
BLOCKED_CONNECTION_TIMEOUT = 600


class Settings:
    cdr_api_token: str
    cdr_host: str
    cogdir: str
    workdir: str
    imagedir: str
    output: str
    callback_secret: str
    callback_url: str
    registration_id: Dict[str, str] = {}
    rabbitmq_host: str
    sequence: List[str] = []


@app.route("/process_event", methods=["POST"])
def process_cdr_event():
    logger.info("event callback started")
    evt = request.get_json(force=True)
    logger.info(f"event data received {evt['event']}")

    map_event: Optional[MapEventPayload] = None
    try:
        # handle event directly or create lara request
        match evt["event"]:
            case "ping":
                logger.info("received ping event")
            case "map.process":
                logger.info("Received map event")
                map_event = MapEventPayload.model_validate(evt["payload"])
            case _:
                logger.info(f"received unsupported {evt['event']} event")

    except Exception:
        logger.error(f"exception processing {evt['event']} event")
        raise

    if not map_event:
        # assume ping or ignored event type
        return Response({"ok": "success"}, status=200, mimetype="application/json")

    first_task = settings.sequence[0]
    first_queue = ChainingResultSubscriber.PIPELINE_QUEUES[first_task]
    first_request = ChainingResultSubscriber.next_request(
        first_task, map_event.cog_id, map_event.cog_url
    )
    request_publisher.publish_lara_request(first_request, first_queue)

    return Response({"ok": "success"}, status=200, mimetype="application/json")


def process_image(image_id: str, request_publisher: LaraRequestPublisher):
    logger.info(f"processing image with id {image_id}")
    image_url = f"{settings.cogdir}/{image_id}.cog.tif"

    # push the request onto the queue
    first_task = settings.sequence[0]
    first_queue = ChainingResultSubscriber.PIPELINE_QUEUES[first_task]
    first_request = ChainingResultSubscriber.next_request(
        first_task, image_id, image_url
    )
    request_publisher.publish_lara_request(first_request, first_queue)


def register_cdr_system():

    for i, pipeline in enumerate(settings.sequence):
        system_name = ChainingResultSubscriber.PIPELINE_SYSTEM_NAMES[pipeline]
        system_version = ChainingResultSubscriber.PIPELINE_SYSTEM_VERSIONS[pipeline]
        logger.info(f"registering system {system_name} with cdr")
        headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}

        # register for all events on the first pipeline others can ignore
        events: Optional[List[str]] = [] if i == 0 else ["ping"]

        registration = {
            "name": system_name,
            "version": system_version,
            "callback_url": settings.callback_url,
            "webhook_secret": settings.callback_secret,
            # Leave blank if callback url has no auth requirement
            # "auth_header": "",
            # "auth_token": "",
            "events": events,
        }

        client = httpx.Client(follow_redirects=True)

        r = client.post(
            f"{settings.cdr_host}/user/me/register", json=registration, headers=headers
        )
        # check if the request was successful
        if r.status_code != 200:
            logger.error(f"failed to register system {system_name} with cdr")
            logger.error(f"response: {r.text}")
            exit(1)

        # Log our registration_id such we can delete it when we close the program.
        response_raw = r.json()
        settings.registration_id[pipeline] = response_raw["id"]
        logger.info(f"system {system_name} registered with cdr")


def get_cdr_registrations() -> List[Dict[str, Any]]:
    logger.info("getting list of existing registrations in CDR")

    # query the listing endpoint in CDR
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    response = client.get(
        f"{settings.cdr_host}/user/me/registrations",
        headers=headers,
    )

    # parse json response
    return json.loads(response.content)


def cdr_unregister(registration_id: str):
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    client.delete(
        f"{settings.cdr_host}/user/me/register/{registration_id}",
        headers=headers,
    )


def cdr_clean_up():
    logger.info(f"unregistering system {settings.registration_id} with cdr")
    # delete our registered system at CDR on program end
    for pipeline in settings.sequence:
        cdr_unregister(settings.registration_id[pipeline])
        logger.info(f"system {settings.registration_id} no longer registered with cdr")


def cdr_startup(host: str):
    # check if already registered and delete existing registrations for this name and token combination
    registrations = get_cdr_registrations()
    if len(registrations) > 0:
        for r in registrations:
            for pipeline in settings.sequence:
                if (
                    r["name"]
                    == ChainingResultSubscriber.PIPELINE_SYSTEM_NAMES[pipeline]
                ):
                    logger.info(f"unregistering system {r['name']} with cdr")
                    cdr_unregister(r["id"])
                    break

    # make it accessible from the outside
    settings.callback_url = f"{host}/process_event"

    register_cdr_system()

    # wire up the cleanup of the registration
    atexit.register(cdr_clean_up)


def start_app():
    # forward ngrok port
    logger.info("using ngrok to forward ports")
    listener = ngrok.forward(APP_PORT, authtoken_from_env=True)
    cdr_startup(listener.url())

    app.run(host="0.0.0.0", port=APP_PORT)


def main():
    # default log settings
    config_logger(logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("process", "host"), required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--cdr_host", type=str, default=CDR_HOST)
    parser.add_argument("--cogdir", type=str, default=COG_PATH)
    parser.add_argument("--metrics_url", type=str, default="")
    parser.add_argument(
        "--sequence",
        nargs="*",
        default=ChainingResultSubscriber.DEFAULT_PIPELINE_SEQUENCE,
    )
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = p.cdr_host
    settings.cogdir = p.cogdir
    settings.callback_secret = CDR_CALLBACK_SECRET
    settings.sequence = p.sequence

    # check parameter consistency: either the mode is process and a cog id is supplied or the mode is host without a cog id
    if p.mode == "process":
        if (p.cog_id == "" or p.cog_id is None) and (p.input == "" or p.input is None):
            logger.info("process mode requires a cog id or an input file")
            exit(1)
    elif p.mode == "host" and (not p.cog_id == "" and p.cog_id is not None):
        logger.info("a cog id cannot be provided if host mode is selected")
        exit(1)
    logger.info(f"starting cdr in {p.mode} mode")

    # declare a global request publisher since we need to access it from the
    # CDR event endpoint
    global request_publisher
    request_publisher = LaraRequestPublisher(
        [
            SEGMENTATION_REQUEST_QUEUE,
            POINTS_REQUEST_QUEUE,
            GEO_REFERENCE_REQUEST_QUEUE,
            METADATA_REQUEST_QUEUE,
        ],
        host=p.host,
        port=p.rabbit_port,
        vhost=p.rabbit_vhost,
        uid=p.rabbit_uid,
        pwd=p.rabbit_pwd,
    )
    request_publisher.start_lara_request_queue()

    # start the listener for the results
    result_subscriber = ChainingResultSubscriber(
        request_publisher,
        LARA_RESULT_QUEUE_NAME,
        settings.cdr_host,
        settings.cdr_api_token,
        host=p.host,
        port=p.rabbit_port,
        vhost=p.rabbit_vhost,
        uid=p.rabbit_uid,
        pwd=p.rabbit_pwd,
        pipeline_sequence=settings.sequence,
    )
    result_subscriber.start_lara_result_queue()

    # either start the flask app if host mode selected or run the image specified if in process mode
    if p.mode == "host":
        start_app()
    elif p.mode == "process":
        cdr_startup("https://mock.example")
        if p.input:
            # open the cog csv file and process each line
            with open(p.input, "r") as f:
                for line in f:
                    cog_id = line.strip()
                    process_image(cog_id, request_publisher)
                    if p.metrics_url != "":
                        requests.post(p.metrics_url + "/counter/jobs_submitted?step=1")
        else:
            process_image(p.cog_id, request_publisher)


if __name__ == "__main__":
    main()
