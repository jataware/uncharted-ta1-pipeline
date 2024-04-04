import argparse
import atexit
import httpx
import json
import logging
import ngrok
import os
import pika
import rasterio as rio
import rasterio.transform as riot
import threading

from flask import Flask, request, Response
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import spec
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

from process.queue import (
    Request,
    RequestResult,
    LARA_REQUEST_QUEUE_NAME,
    LARA_RESULT_QUEUE_NAME,
)

from schema.cdr_schemas.events import Event, MapEventPayload
from schema.cdr_schemas.georeference import GeoreferenceResults

from typing import Any, Dict, List, Optional

logger = logging.getLogger("cdr")

app = Flask(__name__)

request_channel: Optional[Channel] = None

CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]
CDR_HOST = "https://api.cdr.land"
CDR_SYSTEM_NAME = "uncharted"
CDR_SYSTEM_VERSION = "0.0.1"
CDR_CALLBACK_SECRET = "maps rock"
APP_PORT = 5001


class Settings:
    cdr_api_token: str
    cdr_host: str
    workdir: str
    system_name: str
    system_version: str
    callback_secret: str
    callback_url: str
    registration_id: str


settings: Settings


def create_channel(host: str) -> Channel:
    logger.info(f"creating channel on host {host}")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host))
    return connection.channel()


def queue_event(req: Request):
    request_channel = create_channel("localhost")
    request_channel.queue_declare(queue=LARA_REQUEST_QUEUE_NAME)

    logger.info(f"sending request {req.id} for image {req.image_id} to lara queue")
    # send request to queue
    request_channel.basic_publish(
        exchange="",
        routing_key=LARA_REQUEST_QUEUE_NAME,
        body=json.dumps(req.model_dump()),
    )
    logger.info(f"request {req.id} published to lara queue")


def project_image(
    source_image_path: str, target_image_path: str, geo_transform: Affine, crs: str
):
    with rio.open(source_image_path) as raw:
        bounds = riot.array_bounds(raw.height, raw.width, geo_transform)
        pro_transform, pro_width, pro_height = calculate_default_transform(
            crs, crs, raw.width, raw.height, *tuple(bounds)
        )
        pro_kwargs = raw.profile.copy()
        pro_kwargs.update(
            {
                "driver": "COG",
                "crs": {"init": crs},
                "transform": pro_transform,
                "width": pro_width,
                "height": pro_height,
            }
        )
        _raw_data = raw.read()
        with rio.open(target_image_path, "w", **pro_kwargs) as pro:
            for i in range(raw.count):
                _ = reproject(
                    source=_raw_data[i],
                    destination=rio.band(pro, i + 1),
                    src_transform=geo_transform,
                    src_crs=crs,
                    dst_transform=pro_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    num_threads=8,
                    warp_mem_limit=256,
                )


def cps_to_transform(gcps: List[Dict[str, Any]], height: int, to_crs: str) -> Affine:
    cps = [
        {
            "row": height - float(gcp["px_geom"]["rows_from_top"]),
            "col": float(gcp["px_geom"]["columns_from_left"]),
            "x": float(gcp["map_geom"]["longitude"]),
            "y": float(gcp["map_geom"]["latitude"]),
            "crs": gcp["crs"],
        }
        for gcp in gcps
    ]
    cps_p = []
    for cp in cps:
        proj = Transformer.from_crs(cp["crs"], to_crs, always_xy=True)
        x_p, y_p = proj.transform(xx=cp["x"], yy=cp["y"])
        cps_p.append(
            riot.GroundControlPoint(row=cp["row"], col=cp["col"], x=x_p, y=y_p)
        )

    return riot.from_gcps(cps_p)


def project_georeference(
    source_image_path: str,
    target_image_path: str,
    target_crs: str,
    gcps: List[Dict[str, Any]],
):
    # open the image
    img = Image.open(source_image_path)
    _, height = img.size

    # create the transform
    geo_transform = cps_to_transform(gcps, height=height, to_crs=target_crs)

    # use the transform to project the image
    project_image(source_image_path, target_image_path, geo_transform, target_crs)


@app.route("/process_event", methods=["POST"])
def process_event():
    logger.info("event callback started")
    evt = request.get_json(force=True)
    logger.info(f"event data received {evt}")
    lara_req = None

    try:
        # handle event directly or create lara request
        match evt["event"]:
            case "ping":
                logger.info("received ping event")
            case "map.process":
                logger.info("Received map event")
                map_event = MapEventPayload.model_validate(evt["payload"])
                lara_req = Request(
                    id=evt["id"],
                    task="georeference",
                    image_id=map_event.cog_id,
                    image_url=map_event.cog_url,
                    output_format="cdr",
                )
            case _:
                logger.info(f"received unsupported {evt} event")

    except Exception:
        logger.error(f"exception processing {evt} event")
        raise

    if lara_req is None:
        # assume ping or ignored event type
        return Response({"ok": "success"}, status=200, mimetype="application/json")

    # queue event in background since it may be blocking on the queue
    # assert request_channel is not None
    queue_event(lara_req)
    return Response({"ok": "success"}, status=200, mimetype="application/json")


def process_image(image_id: str):
    logger.info(f"processing image with id {image_id}")

    # build the request
    req = Request(
        id="mock",
        task="georeference",
        image_id=image_id,
        image_url=f"https://s3.amazonaws.com/public.cdr.land/cogs/{image_id}.cog.tif",
        output_format="cdr",
    )

    # push the request onto the queue
    queue_event(req)


def process_result(
    channel: Channel,
    method: spec.Basic.Deliver,
    properties: spec.BasicProperties,
    body: bytes,
):
    # ack the result which will result in dropped messages in cases of errors in the processing but prevents blocking on bad data
    channel.basic_ack(delivery_tag=method.delivery_tag)

    logger.info("received data from result channel")
    # parse the result
    body_decoded = json.loads(body.decode())
    result = RequestResult.model_validate(body_decoded)
    logger.info(f"processing result for request {result.request.id}")

    # reproject image to file on disk for pushing to CDR
    georef_result = json.loads(result.output)[0]

    # validate the result by building the model classes
    try:
        GeoreferenceResults.model_validate(georef_result)
    except:
        logger.error(
            "bad georeferencing result received so unable to send results to cdr"
        )
        raise

    projection = georef_result["georeference_results"][0]["projections"][0]
    gcps = georef_result["gcps"]
    output_file_name = projection["file_name"]
    output_file_name_full = os.path.join(settings.workdir, output_file_name)
    logger.info(
        f"projecting image {result.image_path} to {output_file_name_full} using crs {projection['crs']}"
    )
    project_georeference(
        result.image_path, output_file_name_full, projection["crs"], gcps
    )

    files_ = []
    files_.append(("files", (output_file_name, open(output_file_name_full, "rb"))))

    # push the result to CDR
    logger.info(f"pushing result for request {result.request.id} to CDR")
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    resp = client.post(
        f"{settings.cdr_host}/v1/maps/publish/georef",
        data={"georef_result": json.dumps(georef_result)},
        files=files_,
        headers=headers,
    )
    logger.info(
        f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
    )


def start_result_listener(result_queue: str):
    logger.info(
        f"starting the listener on the result queue ({result_queue}:{LARA_RESULT_QUEUE_NAME})"
    )
    # setup the result queue
    result_channel = create_channel(result_queue)
    result_channel.queue_declare(queue=LARA_RESULT_QUEUE_NAME)

    # start consuming the results
    result_channel.basic_qos(prefetch_count=1)
    result_channel.basic_consume(
        queue=LARA_RESULT_QUEUE_NAME, on_message_callback=process_result
    )
    result_channel.start_consuming()


def register_system():
    logger.info("registering system with cdr")
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}

    registration = {
        "name": settings.system_name,
        "version": settings.system_version,
        "callback_url": settings.callback_url,
        "webhook_secret": settings.callback_secret,
        # Leave blank if callback url has no auth requirement
        # "auth_header": "",
        # "auth_token": "",
        # Registers for ALL events
        "events": [],
    }

    client = httpx.Client(follow_redirects=True)

    r = client.post(
        f"{settings.cdr_host}/user/me/register", json=registration, headers=headers
    )

    # Log our registration_id such we can delete it when we close the program.
    settings.registration_id = r.json()["id"]
    logger.info("system registered with cdr")


def get_registrations() -> List[Dict[str, Any]]:
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


def unregister(registration_id: str):
    headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
    client = httpx.Client(follow_redirects=True)
    client.delete(
        f"{settings.cdr_host}/user/me/register/{registration_id}",
        headers=headers,
    )


def clean_up():
    logger.info("unregistering system with cdr")
    # delete our registered system at CDR on program end
    unregister(settings.registration_id)
    logger.info("system no longer registered with cdr")


def start_app():
    # check if already registered and delete existing registrations for this name and token combination
    registrations = get_registrations()
    if len(registrations) > 0:
        for r in registrations:
            if r["name"] == settings.system_name:
                unregister(r["id"])

    # make it accessible from the outside
    listener = ngrok.forward(APP_PORT, authtoken_from_env=True)
    settings.callback_url = listener.url() + "/process_event"

    register_system()

    # wire up the cleanup of the registration
    atexit.register(clean_up)

    app.run(host="0.0.0.0", port=APP_PORT)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--request_queue", type=str, default="localhost")
    parser.add_argument("--result_queue", type=str, default="localhost")
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = CDR_HOST
    settings.workdir = p.workdir
    settings.system_name = CDR_SYSTEM_NAME
    settings.system_version = CDR_SYSTEM_VERSION
    settings.callback_secret = CDR_CALLBACK_SECRET

    # check parameter consistency: either the mode is process and a cog id is supplied or the mode is host without a cog id
    if p.mode == "process" and (p.cog_id == "" or p.cog_id is None):
        logger.info("process mode requires a cog id")
        exit(1)
    elif p.mode == "host" and (not p.cog_id == "" and p.cog_id is not None):
        logger.info("a cog id cannot be provided if host mode is selected")
        exit(1)
    logger.info(f"starting cdr in {p.mode} mode")

    # initializae the request channel
    # global request_channel
    # request_channel = create_channel(p.request_queue)
    # request_channel.queue_declare(queue=LARA_REQUEST_QUEUE_NAME)

    # start the listener for the results
    threading.Thread(target=start_result_listener, args=(p.result_queue,)).start()

    # either start the flask app if host mode selected or run the image specified if in process mode
    if p.mode == "host":
        start_app()
    elif p.mode == "process":
        process_image(p.cog_id)

    # TODO: ensure propoer closure of channels


if __name__ == "__main__":
    main()
