import argparse
import atexit
from pathlib import Path
import httpx
import json
import logging
import ngrok
import os

from flask import Flask, request, Response

from cdr.request_publisher import LaraRequestPublisher
from cdr.result_subscriber import LaraResultSubscriber
from tasks.common.io import download_file
from tasks.common.queue import (
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
CDR_CALLBACK_SECRET = "maps rock"
APP_PORT = 5001
CDR_EVENT_LOG = "events.log"

LARA_RESULT_QUEUE_NAME = "lara_result_queue"

REQUEUE_LIMIT = 3
INACTIVITY_TIMEOUT = 5
HEARTBEAT_INTERVAL = 900
BLOCKED_CONNECTION_TIMEOUT = 600


class Settings:
    cdr_api_token: str
    cdr_host: str
    workdir: str
    imagedir: str
    output: str
    callback_secret: str
    callback_url: str
    registration_id: Dict[str, str] = {}
    rabbitmq_host: str
    serial: bool
    sequence: List[str] = []


def prefetch_image(working_dir: Path, image_id: str, image_url: str) -> None:
    """
    Prefetches the image from the CDR for use by the pipelines.
    """
    # check working dir for the image
    filename = working_dir / f"{image_id}.tif"

    if not os.path.exists(filename):
        # download image
        image_data = download_file(image_url)

        # write it to working dir, creating the directory if necessary
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as file:
            file.write(image_data)


@app.route("/process_event", methods=["POST"])
def process_cdr_event():
    logger.info("event callback started")
    evt = request.get_json(force=True)
    logger.info(f"event data received {evt['event']}")
    lara_reqs: Dict[str, Request] = {}

    try:
        # handle event directly or create lara request
        match evt["event"]:
            case "ping":
                logger.info("received ping event")
            case "map.process":
                logger.info("Received map event")
                map_event = MapEventPayload.model_validate(evt["payload"])
                lara_reqs[GEO_REFERENCE_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="georeference",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[POINTS_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="points",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[SEGMENTATION_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="segments",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
                lara_reqs[METADATA_REQUEST_QUEUE] = Request(
                    id=evt["id"],
                    task="metadata",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
            case _:
                logger.info(f"received unsupported {evt['event']} event")

    except Exception:
        logger.error(f"exception processing {evt['event']} event")
        raise

    if len(lara_reqs) == 0:
        # assume ping or ignored event type
        return Response({"ok": "success"}, status=200, mimetype="application/json")

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.imagedir), map_event.cog_id, map_event.cog_url)

    if settings.serial:
        first_task = settings.sequence[0]
        first_queue = LaraResultSubscriber.PIPELINE_QUEUES[first_task]
        first_request = LaraResultSubscriber.next_request(
            first_task, map_event.cog_id, map_event.cog_url
        )
        request_publisher.publish_lara_request(first_request, first_queue)
    else:
        for queue_name, lara_req in lara_reqs.items():
            request_publisher.publish_lara_request(lara_req, queue_name)

    return Response({"ok": "success"}, status=200, mimetype="application/json")


def process_image(image_id: str, request_publisher: LaraRequestPublisher):
    logger.info(f"processing image with id {image_id}")

    image_url = f"https://s3.amazonaws.com/public.cdr.land/cogs/{image_id}.cog.tif"

    # build the request
    lara_reqs: Dict[str, Request] = {}
    lara_reqs[GEO_REFERENCE_REQUEST_QUEUE] = Request(
        id="mock-georeference",
        task="georeference",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[POINTS_REQUEST_QUEUE] = Request(
        id="mock-points",
        task="points",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[SEGMENTATION_REQUEST_QUEUE] = Request(
        id="mock-segments",
        task="segments",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[METADATA_REQUEST_QUEUE] = Request(
        id="mock-metadata",
        task="metadata",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.imagedir), image_id, image_url)

    # push the request onto the queue
    if settings.serial:
        first_task = settings.sequence[0]
        first_queue = LaraResultSubscriber.PIPELINE_QUEUES[first_task]
        first_request = LaraResultSubscriber.next_request(
            first_task, image_id, image_url
        )
        request_publisher.publish_lara_request(first_request, first_queue)
    else:
        for queue_name, request in lara_reqs.items():
            logger.info(
                f"publishing request for image {image_id} to {queue_name} task: {request.task}"
            )
            request_publisher.publish_lara_request(request, queue_name)


def register_cdr_system():

    for i, pipeline in enumerate(settings.sequence):
        system_name = LaraResultSubscriber.PIPELINE_SYSTEM_NAMES[pipeline]
        system_version = LaraResultSubscriber.PIPELINE_SYSTEM_VERSIONS[pipeline]
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
                if r["name"] == LaraResultSubscriber.PIPELINE_SYSTEM_NAMES[pipeline]:
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
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--imagedir", type=str, required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--cdr_event_log", type=str, default=CDR_EVENT_LOG)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--sequence", nargs="*", default=LaraResultSubscriber.DEFAULT_PIPELINE_SEQUENCE
    )
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = CDR_HOST
    settings.workdir = p.workdir
    settings.imagedir = p.imagedir
    settings.output = p.output
    settings.callback_secret = CDR_CALLBACK_SECRET
    settings.serial = True
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
    publisher = request_publisher if settings.serial else None
    result_subscriber = LaraResultSubscriber(
        publisher,
        LARA_RESULT_QUEUE_NAME,
        settings.cdr_host,
        settings.cdr_api_token,
        settings.output,
        settings.workdir,
        host=p.host,
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
        else:
            process_image(p.cog_id, request_publisher)


if __name__ == "__main__":
    main()
