import argparse
import atexit
import httpx
import json
import logging
import ngrok
import os
import requests

from datetime import datetime
from typing import Any, Dict, List, Optional
from util.logging import config_logger

from flask import Flask, request, Response

from cdr.chaining_result_subscriber import ChainingResultSubscriber
from cdr.request_publisher import LaraRequestPublisher
from tasks.common.request_client import (
    GEO_REFERENCE_REQUEST_QUEUE,
    METADATA_REQUEST_QUEUE,
    POINTS_REQUEST_QUEUE,
    SEGMENTATION_REQUEST_QUEUE,
)

from schema.cdr_schemas.events import MapEventPayload

logger = logging.getLogger("cdr")

app = Flask(__name__)

# Default CDR values
DEFAULT_CDR_HOST = "https://api.cdr.land"
DEFAULT_COG_HOST = "https://s3.amazonaws.com/public.cdr.land/cogs"
DEFAULT_CDR_CALLBACK_SECRET = "maps rock"

# CDR secrets
CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]

APP_PORT = 5001

LARA_RESULT_QUEUE_NAME = "lara_result_queue"

REQUEUE_LIMIT = 3
INACTIVITY_TIMEOUT = 5
HEARTBEAT_INTERVAL = 900
BLOCKED_CONNECTION_TIMEOUT = 600


class Settings:
    cdr_api_token: str
    cdr_host: str
    cog_host: str
    workdir: str
    imagedir: str
    output: str
    callback_secret: str
    callback_url: str
    registration_id: Dict[str, str] = {}
    rabbitmq_host: str
    sequence: List[str] = []
    replay_start: Optional[datetime]
    replay_end: Optional[datetime]


@app.route("/process_event", methods=["POST"])
def process_cdr_event():
    """
    Processes a CDR event received via an HTTP request.
    The function handles different types of events specified in the
    `evt["event"]` field. Supported events include "ping" and
    "map.process". For "map.process" events, it validates the payload
    and initiates a sequence of tasks by publishing a request to a
    message queue. Logs are generated for each step of the process,
    including the start of the event callback, the type of event
    received, and any exceptions that occur during processing.

    Returns:
        Response: A Response object with a JSON payload indicating
        success.
    """

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
    """
    Processes an image by its ID and publishes a request to the appropriate queue.

    Args:
        image_id (str): The ID of the image to be processed.
        request_publisher (LaraRequestPublisher): An instance of LaraRequestPublisher to
            publish the request.

    Returns:
        None
    """

    logger.info(f"processing image with id {image_id}")
    image_url = f"{settings.cog_host}/{image_id}.cog.tif"

    # push the request onto the queue
    first_task = settings.sequence[0]
    first_queue = ChainingResultSubscriber.PIPELINE_QUEUES[first_task]
    first_request = ChainingResultSubscriber.next_request(
        first_task, image_id, image_url
    )
    request_publisher.publish_lara_request(first_request, first_queue)


def register_cdr_system():
    """
    Registers the CDR system with the specified settings.
    This function iterates over the sequence of pipelines defined in the settings
    and registers each pipeline system with the CDR. The first pipeline is registered
    for all events, while subsequent pipelines are registered only for the "ping" event.
    The registration details include the system name, version, callback URL, webhook secret,
    and events. If the registration request fails, the function logs an error message and
    exits the program. Upon successful registration, the registration ID is stored in the
    settings for future reference.

    Raises:
        SystemExit: If the registration request fails.

    Logs:
        Info: When a system is being registered and upon successful registration.
        Error: If the registration request fails.

    Returns:
        None
    """

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
    """
    Fetches the list of existing registrations from the CDR.

    This function sends a GET request to the CDR listing endpoint using an authorization
    token and retrieves the list of registrations associated with the current user.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing registration details.

    Raises:
        httpx.HTTPStatusError: If the request to the CDR endpoint fails.
    """
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
    """
    Unregister a user from the CDR service.
    This function sends a DELETE request to the CDR service to unregister a user
    based on the provided registration ID.
    Args:
        registration_id (str): The unique identifier for the user's registration.
    Raises:
        httpx.HTTPStatusError: If the request to the CDR service fails.
    """

    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    client.delete(
        f"{settings.cdr_host}/user/me/register/{registration_id}",
        headers=headers,
    )


def cdr_clean_up():
    """
    Clean up the CDR by unregistering the system.
    This function logs the process of unregistering the system identified by
    `settings.registration_id` from the CDR. It iterates through the pipelines
    specified in `settings.sequence` and unregisters each one.
    Logging:
        Logs the start and completion of the unregistration process for each pipeline.
    """

    logger.info(f"unregistering system {settings.registration_id} with cdr")
    # delete our registered system at CDR on program end
    for pipeline in settings.sequence:
        cdr_unregister(settings.registration_id[pipeline])
        logger.info(f"system {settings.registration_id} no longer registered with cdr")


def cdr_resend_recent_events(start_time: datetime, end_time: Optional[datetime] = None):
    """
    Resend recent 'map.process' events from the CDR system within a specified
    time range.

    This function fetches notifications for a specific system and version,
    filters the events based on the provided start and end times, and resends
    the filtered notifications.

    Args:
        start_time (datetime): The start time to filter events.
        end_time (Optional[datetime], optional): The end time to filter events.
        Defaults to the current time if not provided.
    """

    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}

    first_system = ChainingResultSubscriber.PIPELINE_SYSTEM_NAMES[settings.sequence[0]]
    first_version = ChainingResultSubscriber.PIPELINE_SYSTEM_VERSIONS[
        settings.sequence[0]
    ]

    # fetch the notifications for this system
    client = httpx.Client(follow_redirects=True)
    response = client.get(
        f"{settings.cdr_host}/user/me/notifications/{first_system}?version={first_version}",
        headers=headers,
    )
    # validate the request was successful
    if response.status_code != 200:
        logger.error("failed to fetch events from cdr")
        logger.error(f"response: {response.text}")
        return []
    response = json.loads(response.content)

    if not end_time:
        end_time = datetime.now()

    # get map.process events between start_time and end_time
    notification_ids: list[str] = [
        e["id"]
        for e in response
        if e["event"]["event"] == "map.process"
        and start_time
        <= datetime.strptime(e["event"]["created_date"], "%Y-%m-%dT%H:%M:%S.%f")
        <= end_time
    ]

    logger.info(
        f"resending {len(notification_ids)} events for {first_system} {first_version} between {start_time} and {end_time}"
    )

    # resend the notifications
    for id in notification_ids:
        response = client.put(
            f"{settings.cdr_host}/user/me/resend/notification/{id}",
            headers=headers,
        )
        # verify the request was successful
        if response.status_code != 200:
            logger.error(f"failed to resend {id}")
            logger.error(f"response: {response.text}")


def cdr_startup(host: str):
    """
    Initialize the CDR system startup process.
    This function performs the following tasks:
    1. Checks for existing CDR registrations and unregisters them if they
       match the current pipeline system names.
    2. Sets the callback URL to make the system accessible from the outside.
    3. Registers the CDR system.
    4. Registers a cleanup function to be called upon program exit.
    5. Fetches recent events if replay settings are configured.

    Args:
        host (str): The host URL to be used for the callback URL.
    """

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

    # fetch recent events
    if settings.replay_start:
        cdr_resend_recent_events(settings.replay_start, settings.replay_end)


def start_app():
    # forward ngrok port
    logger.info("using ngrok to forward ports")
    listener = ngrok.forward(APP_PORT, authtoken_from_env=True)
    cdr_startup(listener.url())

    app.run(host="0.0.0.0", port=APP_PORT)


def valid_date(s) -> datetime:
    """
    Validate a date string in the format 'YYYY-MM-DD-HH:MM:SS

    Args:
        s: The date string to validate.

    Returns:
        The parsed date.
    """
    try:
        return datetime.strptime(s, "%Y-%m-%d-%H:%M:%S")
    except ValueError:
        msg = f"Not a valid date: '{s}'. Expected format: 'YYYY-MM-DD-HH-MM-SS'"
        raise argparse.ArgumentTypeError(msg)


def main():
    # default log settings
    config_logger(logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("process", "host"), required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--cdr_host", type=str, default=DEFAULT_CDR_HOST)
    parser.add_argument("--cog_host", type=str, default=DEFAULT_COG_HOST)
    parser.add_argument(
        "--cdr_callback_secret", type=str, default=DEFAULT_CDR_CALLBACK_SECRET
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--rabbit_port", type=int, default=5672)
    parser.add_argument("--rabbit_vhost", type=str, default="/")
    parser.add_argument("--rabbit_uid", type=str, default="")
    parser.add_argument("--rabbit_pwd", type=str, default="")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--metrics_url", type=str, default="")
    parser.add_argument(
        "--sequence",
        nargs="*",
        default=ChainingResultSubscriber.DEFAULT_PIPELINE_SEQUENCE,
    )
    parser.add_argument("--replay_start", type=valid_date, required=False)
    parser.add_argument("--replay_end", type=valid_date, required=False)
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = p.cdr_host
    settings.cog_host = p.cog_host
    settings.callback_secret = p.cdr_callback_secret
    settings.sequence = p.sequence
    settings.replay_start = p.replay_start if hasattr(p, "replay_start") else None
    settings.replay_end = p.replay_end if hasattr(p, "replay_end") else None

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
