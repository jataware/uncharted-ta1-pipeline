import json
import logging
import os
import pprint
import threading
from time import sleep
from typing import List, Optional
import httpx
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
import rasterio as rio
import rasterio.transform as riot
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from pika.exceptions import AMQPChannelError, AMQPConnectionError
import pika.spec as spec
from pydantic import BaseModel
from regex import P
from cdr.request_publisher import LaraRequestPublisher
from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.georeference import GeoreferenceResults, GroundControlPoint
from schema.cdr_schemas.metadata import CogMetaData
from tasks.common.queue import (
    GEO_REFERENCE_REQUEST_QUEUE,
    METADATA_REQUEST_QUEUE,
    POINTS_REQUEST_QUEUE,
    REQUEUE_LIMIT,
    SEGMENTATION_REQUEST_QUEUE,
    OutputType,
    Request,
    RequestResult,
)
from schema.mappers.cdr import GeoreferenceMapper, get_mapper
from tasks.geo_referencing.entities import (
    GeoreferenceResult as LARAGeoreferenceResult,
    GroundControlPoint as LARAGroundControlPoint,
)
from tasks.geo_referencing.util import cps_to_transform, project_image
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata
from tasks.point_extraction.entities import PointLabels as LARAPoints
from tasks.segmentation.entities import MapSegmentation as LARASegmentation
import datetime

logger = logging.getLogger("result_subscriber")


class LaraResultSubscriber:

    REQUEUE_LIMIT = 3
    INACTIVITY_TIMEOUT = 5
    HEARTBEAT_INTERVAL = 900
    BLOCKED_CONNECTION_TIMEOUT = 600

    # pipeline name definitions
    SEGMENTATION_PIPELINE = "segmentation"
    METADATA_PIPELINE = "metadata"
    POINTS_PIPELINE = "points"
    GEOREFERENCE_PIPELINE = "georeference"
    NULL_PIPELINE = "null"

    # pipeline related rabbitmq queue names
    PIPELINE_QUEUES = {
        SEGMENTATION_PIPELINE: SEGMENTATION_REQUEST_QUEUE,
        METADATA_PIPELINE: METADATA_REQUEST_QUEUE,
        POINTS_PIPELINE: POINTS_REQUEST_QUEUE,
        GEOREFERENCE_PIPELINE: GEO_REFERENCE_REQUEST_QUEUE,
    }

    # map of pipeline output types to pipeline names
    PIPELINE_OUTPUTS = {
        OutputType.SEGMENTATION: SEGMENTATION_PIPELINE,
        OutputType.METADATA: METADATA_PIPELINE,
        OutputType.POINTS: POINTS_PIPELINE,
        OutputType.GEOREFERENCING: GEOREFERENCE_PIPELINE,
    }

    # sequence of pipelines execution
    DEFAULT_PIPELINE_SEQUENCE = [
        SEGMENTATION_PIPELINE,
        METADATA_PIPELINE,
        POINTS_PIPELINE,
        GEOREFERENCE_PIPELINE,
    ]

    # map of pipeline name to system name
    PIPELINE_SYSTEM_NAMES = {
        SEGMENTATION_PIPELINE: "uncharted-area",
        METADATA_PIPELINE: "uncharted-metadata",
        POINTS_PIPELINE: "uncharted-points",
        GEOREFERENCE_PIPELINE: "uncharted-georeference",
    }

    # map of pipeline name to system version
    PIPELINE_SYSTEM_VERSIONS = {
        SEGMENTATION_PIPELINE: "0.0.5",
        METADATA_PIPELINE: "0.0.5",
        POINTS_PIPELINE: "0.0.5",
        GEOREFERENCE_PIPELINE: "0.0.6",
    }

    def __init__(
        self,
        request_publisher: Optional[LaraRequestPublisher],
        result_queue: str,
        cdr_host: str,
        cdr_token: str,
        output: str,
        workdir: str,
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
        pipeline_sequence: List[str] = DEFAULT_PIPELINE_SEQUENCE,
    ) -> None:
        self._request_publisher = request_publisher
        self._result_connection: Optional[BlockingConnection] = None
        self._result_channel: Optional[Channel] = None
        self._result_queue = result_queue
        self._cdr_host = cdr_host
        self._cdr_token = cdr_token
        self._workdir = workdir
        self._output = output
        self._host = host
        self._port = port
        self._vhost = vhost
        self._uid = uid
        self._pwd = pwd
        self._pipeline_sequence = (
            pipeline_sequence
            if len(pipeline_sequence) > 0
            else self.DEFAULT_PIPELINE_SEQUENCE
        )

    def start_lara_result_queue(self):
        threading.Thread(
            target=self._run_lara_result_queue,
            args=(self._result_queue, self._host),
        ).start()

    def _create_channel(self, host: str, queue: str) -> Channel:
        """
        Creates a blocking connection and channel on the given host and declares the given queue.

        Args:
            host: The host to connect to.
            queue: The queue to declare.

        Returns:
            The created channel.
        """
        logger.info(f"creating channel on host {host}")
        if self._uid != "":
            credentials = PlainCredentials(self._uid, self._pwd)
            connection = BlockingConnection(
                ConnectionParameters(
                    self._host,
                    self._port,
                    self._vhost,
                    credentials,
                    heartbeat=self.HEARTBEAT_INTERVAL,
                    blocked_connection_timeout=self.BLOCKED_CONNECTION_TIMEOUT,
                )
            )
        else:
            connection = BlockingConnection(
                ConnectionParameters(
                    self._host,
                    heartbeat=self.HEARTBEAT_INTERVAL,
                    blocked_connection_timeout=self.BLOCKED_CONNECTION_TIMEOUT,
                )
            )
        channel = connection.channel()
        channel.queue_declare(
            queue=queue,
            durable=True,
            arguments={"x-delivery-limit": REQUEUE_LIMIT, "x-queue-type": "quorum"},
        )
        return channel

    @staticmethod
    def next_request(next_pipeline: str, image_id: str, image_url: str) -> Request:
        """
        Creates a new Request object for the next pipeline.

        Args:
            next_pipeline (str): The name of the next pipeline.
            image_id (str): The ID of the image.
            image_url (str): The URL of the image.

        Returns:
            Request: A new Request object with the specified parameters.
        """
        # Get the current UTC time
        current_time = datetime.datetime.now(datetime.timezone.utc)
        timestamp = int(current_time.timestamp())

        return Request(
            id=f"{next_pipeline}-{timestamp}-pipeline",
            task=f"{next_pipeline}",
            output_format="cdr",
            image_id=image_id,
            image_url=image_url,
        )

    def _process_lara_result(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        properties: spec.BasicProperties,
        body: bytes,
    ):
        """
        Process the received LARA result.  In a serial execution model this will also
        trigger the next pipeline in the sequence.

        Args:
            channel (Channel): The channel object.
            method (spec.Basic.Deliver): The method object.
            properties (spec.BasicProperties): The properties object.
            body (bytes): The body of the message.

        Returns:
            None

        Raises:
            Exception: If there is an error processing the result.
        """
        try:
            logger.info("received data from result channel")
            # parse the result
            body_decoded = json.loads(body.decode())
            result = RequestResult.model_validate(body_decoded)
            logger.info(
                f"processing result for request {result.request.id} of type {result.output_type}"
            )

            match result.output_type:
                case OutputType.SEGMENTATION:
                    logger.info("segmentation results received")
                    self._push_segmentation(result)
                case OutputType.METADATA:
                    logger.info("metadata results received")
                    self._push_metadata(result)
                case OutputType.POINTS:
                    logger.info("points results received")
                    self._push_points(result)
                case OutputType.GEOREFERENCING:
                    logger.info("georeferencing results received")
                    self._push_georeferencing(result)
                case _:
                    logger.info("unsupported output type received from queue")

            # in the serial case we call the next pipeline in the sequence
            if self._request_publisher:
                # find the next pipeline
                output_pipeline = self.PIPELINE_OUTPUTS[result.output_type]
                next = self._pipeline_sequence.index(output_pipeline) + 1
                next_pipeline = (
                    self._pipeline_sequence[next]
                    if next < len(self._pipeline_sequence)
                    else self.NULL_PIPELINE
                )

                # if there is no next pipeline in the sequence then we are done
                if next_pipeline == self.NULL_PIPELINE:
                    return

                request = self.next_request(
                    next_pipeline,
                    result.request.image_id,
                    result.request.image_url,
                )
                logger.info(f"sending next request in sequence: {request.task}")
                self._request_publisher.publish_lara_request(
                    request, self.PIPELINE_QUEUES[next_pipeline]
                )

        except Exception as e:
            logger.exception(f"Error processing result: {str(e)}")

        logger.info("result processing finished")

    def _run_lara_result_queue(self, result_queue: str, host="localhost"):
        """
        Main loop to service the result queue. process_data_events is set to block for a maximum
        of 1 second before returning to ensure that heartbeats etc. are processed.
        """
        while True:
            result_channel: Optional[Channel] = None
            try:
                logger.info(
                    f"starting the listener on the result queue ({host}:{result_queue})"
                )
                # setup the result queue
                result_channel = self._create_channel(host, result_queue)
                result_channel.basic_qos(prefetch_count=1)

                # start consuming the results - will timeout after 5 seconds of inactivity
                # allowing things like heartbeats to be processed
                while True:
                    for method_frame, properties, body in result_channel.consume(
                        result_queue, inactivity_timeout=self.INACTIVITY_TIMEOUT
                    ):
                        if method_frame:
                            try:
                                self._process_lara_result(
                                    result_channel, method_frame, properties, body
                                )
                                result_channel.basic_ack(method_frame.delivery_tag)
                            except Exception as e:
                                logger.exception(e)
                                result_channel.basic_nack(method_frame.delivery_tag)

            except (AMQPConnectionError, AMQPChannelError):
                logger.warning(f"result channel closed, reconnecting")
                # channel is closed - make sure the connection is closed to facilitate a
                # clean reconnect
                if result_channel and not result_channel.connection.is_closed:
                    logger.info("closing result connection")
                    result_channel.connection.close()
                sleep(5)

    def _write_cdr_result(
        self, image_id: str, output_type: OutputType, result: BaseModel
    ):
        """
        Write the CDR result to a JSON file.

        Args:
            image_id (str): The ID of the image.
            output_type (OutputType): The type of output.
            result (BaseModel): The result to be written.

        Returns:
            None
        """
        if self._output:
            output_file = os.path.join(
                self._output,
                f"{image_id}_{output_type.name.lower()}.json",
            )
            os.makedirs(
                self._output, exist_ok=True
            )  # Create the output directory if it doesn't exist
            with open(output_file, "a") as f:
                logger.info(f"writing result to {output_file}")
                f.write(json.dumps(result.model_dump()))
                f.write("\n")
            return

    def _push_georeferencing(self, result: RequestResult):
        # reproject image to file on disk for pushing to CDR
        georef_result_raw = json.loads(result.output)

        # don't write when the result is empty
        if len(georef_result_raw) == 0:
            logger.info(
                f"empty georef result received for {result.request.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[GeoreferenceResults] = None
        files_ = []
        try:
            lara_result = LARAGeoreferenceResult.model_validate(georef_result_raw)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.GEOREFERENCE_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.GEOREFERENCE_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
            assert cdr_result is not None
            assert cdr_result.georeference_results is not None
            assert cdr_result.georeference_results[0] is not None
            assert cdr_result.georeference_results[0].projections is not None
            projection = cdr_result.georeference_results[0].projections[0]
            gcps = cdr_result.gcps
            output_file_name = projection.file_name
            output_file_name_full = os.path.join(self._workdir, output_file_name)

            assert gcps is not None
            lara_gcps = [
                LARAGroundControlPoint(
                    id=f"gcp.{i}",
                    pixel_x=gcp.px_geom.columns_from_left,
                    pixel_y=gcp.px_geom.rows_from_top,
                    latitude=gcp.map_geom.latitude if gcp.map_geom.latitude else 0,
                    longitude=gcp.map_geom.longitude if gcp.map_geom.longitude else 0,
                    confidence=gcp.confidence if gcp.confidence else 0,
                )
                for i, gcp in enumerate(gcps)
            ]

            logger.info(
                f"projecting image {result.image_path} to {output_file_name_full}. CRS: {lara_result.crs} -> {GeoreferenceMapper.DEFAULT_OUTPUT_CRS}"
            )
            self._project_georeference(
                result.image_path,
                output_file_name_full,
                lara_result.crs,
                GeoreferenceMapper.DEFAULT_OUTPUT_CRS,
                lara_gcps,
            )

            files_.append(
                ("files", (output_file_name, open(output_file_name_full, "rb")))
            )
        except Exception as e:
            logger.exception(
                "bad georeferencing result received so creating an empty result to send to cdr",
                e,
            )

            # create an empty result to send to cdr
            cdr_result = GeoreferenceResults(
                cog_id=result.request.image_id,
                georeference_results=[],
                gcps=[],
                system=self.PIPELINE_SYSTEM_NAMES[self.GEOREFERENCE_PIPELINE],
                system_version=self.PIPELINE_SYSTEM_VERSIONS[
                    self.GEOREFERENCE_PIPELINE
                ],
            )

        assert cdr_result is not None
        try:
            # write the result to disk if output is set
            if self._output:
                self._write_cdr_result(
                    result.request.image_id, result.output_type, cdr_result
                )
                return

            # push the result to CDR
            logger.info(f"pushing result for request {result.request.id} to CDR")
            headers = {"Authorization": f"Bearer {self._cdr_token}"}
            client = httpx.Client(follow_redirects=True)
            resp = client.post(
                f"{self._cdr_host}/v1/maps/publish/georef",
                data={"georef_result": json.dumps(cdr_result.model_dump())},
                files=files_,
                headers=headers,
                timeout=None,
            )
            logger.info(
                f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
            )
        except Exception as e:
            logger.exception(
                "error when attempting to submit georeferencing results",
                e,
                exc_info=True,
            )

    def _project_georeference(
        self,
        source_image_path: str,
        target_image_path: str,
        source_crs: str,
        target_crs: str,
        gcps: List[LARAGroundControlPoint],
    ):
        """
        Projects an image to a new coordinate reference system using ground control points.

        Args:
            source_image_path (str): The path to the source image.
            target_image_path (str): The path to the target image.
            target_crs (str): The target coordinate reference system.
            gcps (List[GroundControlPoint]): The ground control points.
        """
        # open the image
        img = Image.open(source_image_path)

        # create the transform and use it to project the image
        geo_transform = cps_to_transform(gcps, source_crs, target_crs)
        image_bytes = project_image(img, geo_transform, target_crs)

        # write the projected image to disk, creating the directory if it doesn't exist
        os.makedirs(os.path.dirname(target_image_path), exist_ok=True)
        with open(target_image_path, "wb") as f:
            f.write(image_bytes.getvalue())

    def _push_features(self, result: RequestResult, model: FeatureResults):
        """
        Pushes the features result to the CDR
        """
        if self._output:
            self._write_cdr_result(result.request.image_id, result.output_type, model)
            return

        logger.info(f"pushing features result for request {result.request.id} to CDR")
        headers = {
            "Authorization": f"Bearer {self._cdr_token}",
            "Content-Type": "application/json",
        }
        client = httpx.Client(follow_redirects=True)
        resp = client.post(
            f"{self._cdr_host}/v1/maps/publish/features",
            data=model.model_dump_json(),  #   type: ignore
            headers=headers,
            timeout=None,
        )
        logger.info(
            f"result for request {result.request.id} sent to CDR with response {resp.status_code}: {resp.content}"
        )

    def _push_segmentation(self, result: RequestResult):
        """
        Pushes the segmentation result to the CDR
        """
        segmentation_raw_result = json.loads(result.output)

        # don't write when the result is empty
        if len(segmentation_raw_result) == 0:
            logger.info(
                f"empty segmentation result received for {result.request.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[FeatureResults] = None
        try:
            lara_result = LARASegmentation.model_validate(segmentation_raw_result)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.SEGMENTATION_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.SEGMENTATION_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except:
            logger.error(
                "bad segmentation result received so unable to send results to cdr"
            )
            return

        assert cdr_result is not None
        self._push_features(result, cdr_result)

    def _push_points(self, result: RequestResult):
        points_raw_result = json.loads(result.output)

        # don't write when the result is empty
        if len(points_raw_result) == 0:
            logger.info(
                f"empty point result received for {result.request.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[FeatureResults] = None
        try:
            lara_result = LARAPoints.model_validate(points_raw_result)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.POINTS_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.POINTS_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except:
            logger.error("bad points result received so unable to send results to cdr")
            return

        assert cdr_result is not None
        self._push_features(result, cdr_result)

    def _push_metadata(self, result: RequestResult):
        """
        Pushes the metadata result to the CDR
        """
        metadata_result_raw = json.loads(result.output)

        # don't write when the result is empty
        if len(metadata_result_raw) == 0:
            logger.info(
                f"empty metadata result received for {result.request.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[CogMetaData] = None
        try:
            lara_result = LARAMetadata.model_validate(metadata_result_raw)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.METADATA_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.METADATA_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except Exception as e:
            logger.exception(
                e,
                f"bad metadata result received for {result.request.image_id} so unable to send results to cdr",
            )
            return

        assert cdr_result is not None

        # wrap metadata into feature result
        final_result = FeatureResults(
            cog_id=result.request.image_id,
            cog_metadata_extractions=[cdr_result],
            system=cdr_result.system,
            system_version=cdr_result.system_version,
        )

        self._push_features(result, final_result)
