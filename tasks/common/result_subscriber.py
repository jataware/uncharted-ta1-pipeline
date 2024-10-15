from abc import ABC, abstractmethod
import logging
import threading
from time import sleep
from typing import Optional
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from pika.exceptions import AMQPChannelError, AMQPConnectionError
import pika.spec as spec
from tasks.common.image_cache import ImageCache
from tasks.common.request_client import (
    REQUEUE_LIMIT,
    Request,
)
import datetime

logger = logging.getLogger("result_subscriber")


class LaraResultSubscriber(ABC):

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
        result_queue: str,
        cdr_host: str,
        cdr_token: str,
        output: str,
        workdir: str,
        imagedir: str,
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
    ) -> None:
        self._result_connection: Optional[BlockingConnection] = None
        self._result_channel: Optional[Channel] = None
        self._result_queue = result_queue
        self._cdr_host = cdr_host
        self._cdr_token = cdr_token
        self._workdir = workdir
        self._imagedir = imagedir
        self._output = output
        self._host = host
        self._port = port
        self._vhost = vhost
        self._uid = uid
        self._pwd = pwd
        self._image_cache = ImageCache(imagedir)
        self._image_cache._init_cache()

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

    @abstractmethod
    def _process_lara_result(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        properties: spec.BasicProperties,
        body: bytes,
    ):
        pass

    def _run_lara_result_queue(self, result_queue: str, host="localhost"):
        """
        Main loop to service the result queue. process_data_events is set to block for a maximum
        of 1 second before returning to ensure that heartbeats etc. are processed.

        Args:
            result_queue (str): The name of the result queue to listen to.
            host (str, optional): The hostname of the message broker. Defaults to "localhost".
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
