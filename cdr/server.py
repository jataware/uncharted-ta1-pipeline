import argparse
import atexit
from pathlib import Path
import httpx
import json
import logging
import coloredlogs
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

from tasks.common.io import ImageFileInputIterator, download_file
from tasks.common.queue import (
    GEO_REFERENCE_REQUEST_QUEUE,
    POINTS_REQUEST_QUEUE,
    SEGMENTATION_REQUEST_QUEUE,
    OutputType,
    Request,
    RequestResult,
)

from schema.mappers.cdr import get_mapper
from schema.cdr_schemas.events import Event, MapEventPayload
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.georeference import GeoreferenceResults, GroundControlPoint
from schema.cdr_schemas.metadata import CogMetaData
from tasks.geo_referencing.coordinates_extractor import RE_DEG
from tasks.geo_referencing.entities import GeoreferenceResult as LARAGeoreferenceResult
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata
from tasks.point_extraction.entities import MapImage as LARAPoints
from tasks.segmentation.entities import MapSegmentation as LARASegmentation

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cdr")

app = Flask(__name__)

request_channel: Optional[Channel] = None

CDR_API_TOKEN = os.environ["CDR_API_TOKEN"]
CDR_HOST = "https://api.cdr.land"
CDR_SYSTEM_NAME = "uncharted-ph"
CDR_SYSTEM_VERSION = "0.0.1"
CDR_CALLBACK_SECRET = "maps rock"
APP_PORT = 5001

LARA_RESULT_QUEUE_NAME = "lara_result_queue"


class Settings:
    cdr_api_token: str
    cdr_host: str
    workdir: str
    system_name: str
    system_version: str
    callback_secret: str
    callback_url: str
    registration_id: str
    rabbitmq_host: str


settings: Settings


def create_channel(host: str) -> Channel:
    logger.info(f"creating channel on host {host}")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host))
    return connection.channel()


def publish_lara_request(req: Request, request_queue: str, host="localhost"):
    request_channel = create_channel(host)
    request_channel.queue_declare(queue=request_queue)

    logger.info(f"sending request {req.id} for image {req.image_id} to lara queue")
    # send request to queue
    request_channel.basic_publish(
        exchange="",
        routing_key=request_queue,
        body=json.dumps(req.model_dump()),
    )
    logger.info(f"request {req.id} published to lara queue")
    request_channel.connection.close()


def prefetch_image(working_dir: Path, image_id: str, image_url: str) -> None:
    """
    Prefetches the image from the CDR for use by the pipelines.
    """
    # check working dir for the image
    filename = working_dir / f"{image_id}.tif"

    if not filename.exists():
        # download image
        image_data = download_file(image_url)

        # write it to working dir, creating the directory if necessary
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as file:
            file.write(image_data)


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


def cps_to_transform(
    gcps: List[GroundControlPoint], height: int, to_crs: str
) -> Affine:
    cps = [
        {
            "row": height - float(gcp.px_geom.rows_from_top),
            "col": float(gcp.px_geom.columns_from_left),
            "x": float(gcp.map_geom.longitude),  #   type: ignore
            "y": float(gcp.map_geom.latitude),  #   type: ignore
            "crs": gcp.crs,
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
    gcps: List[GroundControlPoint],
):
    # open the image
    img = Image.open(source_image_path)
    _, height = img.size

    # create the transform
    geo_transform = cps_to_transform(gcps, height=height, to_crs=target_crs)

    # use the transform to project the image
    project_image(source_image_path, target_image_path, geo_transform, target_crs)


@app.route("/process_event", methods=["POST"])
def process_cdr_event():
    logger.info("event callback started")
    evt = request.get_json(force=True)
    logger.info(f"event data received {evt}")
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

            case _:
                logger.info(f"received unsupported {evt} event")

    except Exception:
        logger.error(f"exception processing {evt} event")
        raise

    if len(lara_reqs) == 0:
        # assume ping or ignored event type
        return Response({"ok": "success"}, status=200, mimetype="application/json")

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.workdir), map_event.cog_id, map_event.cog_url)
    # queue event in background since it may be blocking on the queue
    # assert request_channel is not None
    for queue_name, lara_req in lara_reqs.items():
        publish_lara_request(lara_req, queue_name, settings.rabbitmq_host)

    return Response({"ok": "success"}, status=200, mimetype="application/json")


def process_image(image_id: str):
    logger.info(f"processing image with id {image_id}")

    image_url = f"https://s3.amazonaws.com/public.cdr.land/cogs/{image_id}.cog.tif"

    # build the request
    lara_reqs: Dict[str, Request] = {}
    lara_reqs[GEO_REFERENCE_REQUEST_QUEUE] = Request(
        id="mock",
        task="georeference",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[POINTS_REQUEST_QUEUE] = Request(
        id="mock-pts",
        task="points",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )
    lara_reqs[POINTS_REQUEST_QUEUE] = Request(
        id="mock-segments",
        task="segments",
        image_id=image_id,
        image_url=image_url,
        output_format="cdr",
    )

    # Pre-fetch the image from th CDR for use by the pipelines.  The pipelines have an
    # imagedir arg that should be configured to point at this location.
    prefetch_image(Path(settings.workdir), image_id, image_url)

    # push the request onto the queue
    for queue_name, request in lara_reqs.items():
        publish_lara_request(request, queue_name)


def push_georeferencing(result: RequestResult):
    # reproject image to file on disk for pushing to CDR
    georef_result_raw = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[GeoreferenceResults] = None
    try:
        lara_result = LARAGeoreferenceResult.model_validate(georef_result_raw)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error(
            "bad georeferencing result received so unable to send results to cdr"
        )
        raise

    assert cdr_result is not None
    assert cdr_result.georeference_results is not None
    assert cdr_result.georeference_results[0] is not None
    assert cdr_result.georeference_results[0].projections is not None
    projection = cdr_result.georeference_results[0].projections[0]
    gcps = cdr_result.gcps
    output_file_name = projection.file_name
    output_file_name_full = os.path.join(settings.workdir, output_file_name)

    assert gcps is not None
    try:
        logger.info(
            f"projecting image {result.image_path} to {output_file_name_full} using crs {projection.crs}"
        )
        project_georeference(
            result.image_path, output_file_name_full, projection.crs, gcps
        )

        files_ = []
        files_.append(("files", (output_file_name, open(output_file_name_full, "rb"))))

        # push the result to CDR
        logger.info(f"pushing result for request {result.request.id} to CDR")
        headers = {"Authorization": f"Bearer {settings.cdr_api_token}"}
        client = httpx.Client(follow_redirects=True)
        resp = client.post(
            f"{settings.cdr_host}/v1/maps/publish/georef",
            data={"georef_result": json.dumps(cdr_result.model_dump())},
            files=files_,
            headers=headers,
        )
        logger.info(
            f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
        )
    except:
        logger.info("error when attempting to submit georeferencing results")


def push_features(result: RequestResult, model: FeatureResults):
    """
    Pushes the features result to the CDR
    """
    logger.info(f"pushing features result for request {result.request.id} to CDR")
    headers = {
        "Authorization": f"Bearer {settings.cdr_api_token}",
        "Content-Type": "application/json",
    }
    client = httpx.Client(follow_redirects=True)
    resp = client.post(
        f"{settings.cdr_host}/v1/maps/publish/features",
        data=model.model_dump_json(),  #   type: ignore
        headers=headers,
    )
    logger.info(
        f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
    )


def push_segmentation(result: RequestResult):
    """
    Pushes the segmentation result to the CDR
    """
    segmentation_raw_result = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[FeatureResults] = None
    try:
        lara_result = LARASegmentation.model_validate(segmentation_raw_result)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error(
            "bad segmentation result received so unable to send results to cdr"
        )
        return

    assert cdr_result is not None
    push_features(result, cdr_result)


def push_points(result: RequestResult):
    points_raw_result = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[FeatureResults] = None
    try:
        lara_result = LARAPoints.model_validate(points_raw_result)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error("bad points result received so unable to send results to cdr")
        return

    assert cdr_result is not None
    push_features(result, cdr_result)


def push_metadata(result: RequestResult):
    """
    Pushes the metadata result to the CDR
    """
    metadata_result_raw = json.loads(result.output)

    # validate the result by building the model classes
    cdr_result: Optional[CogMetaData] = None
    try:
        lara_result = LARAMetadata.model_validate(metadata_result_raw)
        mapper = get_mapper(lara_result, settings.system_name, settings.system_version)
        cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
    except:
        logger.error("bad metadata result received so unable to send results to cdr")
        return

    assert cdr_result is not None

    # wrap metadata into feature result
    final_result = FeatureResults(
        cog_id=result.request.image_id,
        cog_metadata_extractions=[cdr_result],
        system=cdr_result.system,
        system_version=cdr_result.system_version,
    )

    push_features(result, final_result)


def process_lara_result(
    channel: Channel,
    method: spec.Basic.Deliver,
    properties: spec.BasicProperties,
    body: bytes,
):
    try:
        logger.info("received data from result channel")
        # parse the result
        body_decoded = json.loads(body.decode())
        result = RequestResult.model_validate(body_decoded)
        logger.info(
            f"processing result for request {result.request.id} of type {result.output_type}"
        )

        # reproject image to file on disk for pushing to CDR
        match result.output_type:
            case OutputType.GEOREFERENCING:
                logger.info("georeferencing results received")
                push_georeferencing(result)
            case OutputType.METADATA:
                logger.info("metadata results received")
                push_metadata(result)
            case OutputType.SEGMENTATION:
                logger.info("segmentation results received")
                push_segmentation(result)
            case OutputType.POINTS:
                logger.info("points results received")
                push_points(result)
            case _:
                logger.info("unsupported output type received from queue")

    except Exception as e:
        logger.exception(f"Error processing result: {str(e)}")

    logger.info("result processing finished")


def start_lara_result_listener(result_queue: str, host="localhost"):
    logger.info(f"starting the listener on the result queue ({host}:{result_queue})")
    # setup the result queue
    result_channel = create_channel(host)
    result_channel.queue_declare(queue=result_queue)

    # start consuming the results
    result_channel.basic_qos(prefetch_count=1)
    result_channel.basic_consume(
        queue=result_queue, on_message_callback=process_lara_result, auto_ack=True
    )
    result_channel.start_consuming()


def register_cdr_system():
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
    response_raw = r.json()
    settings.registration_id = response_raw["id"]
    logger.info("system registered with cdr")


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
    logger.info("unregistering system with cdr")
    # delete our registered system at CDR on program end
    cdr_unregister(settings.registration_id)
    logger.info("system no longer registered with cdr")


def cdr_startup(host: str):
    # check if already registered and delete existing registrations for this name and token combination
    registrations = get_cdr_registrations()
    if len(registrations) > 0:
        for r in registrations:
            if r["name"] == settings.system_name:
                cdr_unregister(r["id"])

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
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"color": "white"}
    coloredlogs.install(logger=logger)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("process", "host"), required=True)
    parser.add_argument("--system", type=str, default=CDR_SYSTEM_NAME)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--cog_id", type=str, required=False)
    parser.add_argument("--host", type=str, default="localhost")
    p = parser.parse_args()

    global settings
    settings = Settings()
    settings.cdr_api_token = CDR_API_TOKEN
    settings.cdr_host = CDR_HOST
    settings.workdir = p.workdir
    settings.system_name = p.system
    settings.system_version = CDR_SYSTEM_VERSION
    settings.callback_secret = CDR_CALLBACK_SECRET
    settings.rabbitmq_host = p.host

    # check parameter consistency: either the mode is process and a cog id is supplied or the mode is host without a cog id
    if p.mode == "process" and (p.cog_id == "" or p.cog_id is None):
        logger.info("process mode requires a cog id")
        exit(1)
    elif p.mode == "host" and (not p.cog_id == "" and p.cog_id is not None):
        logger.info("a cog id cannot be provided if host mode is selected")
        exit(1)
    logger.info(f"starting cdr in {p.mode} mode")

    # start the listener for the results
    threading.Thread(
        target=start_lara_result_listener,
        args=(
            "lara_result_queue",
            p.host,
        ),
    ).start()

    # either start the flask app if host mode selected or run the image specified if in process mode
    if p.mode == "host":
        start_app()
    elif p.mode == "process":
        cdr_startup("https://mock.example")
        process_image(p.cog_id)


if __name__ == "__main__":
    main()
