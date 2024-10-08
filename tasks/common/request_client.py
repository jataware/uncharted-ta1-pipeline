from enum import Enum
import json
import logging
import os
from threading import Thread
from time import sleep
import requests
import time

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from PIL.Image import Image as PILImage

from tasks.common.image_cache import ImageCache
from tasks.common.pipeline import (
    BaseModelOutput,
    Output,
    Pipeline,
    PipelineInput,
)
from tasks.common.io import ImageFileReader
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, spec
from pydantic import BaseModel
from typing import Optional, Tuple

logger = logging.getLogger("request_queue")

# default queue names

METADATA_REQUEST_QUEUE = "metadata_request"
METADATA_RESULT_QUEUE = "metadata_result"

GEO_REFERENCE_REQUEST_QUEUE = "georef_request"
GEO_REFERENCE_RESULT_QUEUE = "georef_result"

SEGMENTATION_REQUEST_QUEUE = "segmentation_request"
SEGMENTATION_RESULT_QUEUE = "segmentation_result"

POINTS_REQUEST_QUEUE = "points_request"
POINTS_RESULT_QUEUE = "points_result"

TEXT_REQUEST_QUEUE = "text_request"
TEXT_RESULT_QUEUE = "text_result"

WRITE_REQUEST_QUEUE = "write_request"

REQUEUE_LIMIT = 3


class Request(BaseModel):
    """
    A request to run an image through a pipeline.
    """

    id: str
    task: str
    image_id: str
    image_url: str
    output_format: str


class OutputType(int, Enum):
    GEOREFERENCING = 1
    METADATA = 2
    SEGMENTATION = 3
    POINTS = 4
    TEXT = 5


class RequestResult(BaseModel):
    """
    The result of a pipeline request.
    """

    request: Request
    success: bool
    output: str
    output_type: OutputType
    image_path: str


class RequestClient:
    """
    RequestClient is a class that handles the processing of requests and results through
    a pipeline. It connects to request and result queues, processes incoming requests,
    runs them through a pipeline, and publishes the results.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        request_queue: str,
        result_queue: str,
        output_key: str,
        output_type: OutputType,
        imagedir: str,
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
        metrics_url="",
        metrics_type="",
        heartbeat=900,
        blocked_connection_timeout=600,
    ) -> None:
        """
        Initializes the RequestClient.
        Args:
            pipeline (Pipeline): The pipeline instance to be used.
            request_queue (str): The name of the request queue.
            result_queue (str): The name of the result queue.
            output_key (str): The key for the output.
            output_type (OutputType): The type of the output.
            imagedir (str): The directory for storing images.
            host (str, optional): The host address for the connection. Defaults to "localhost".
            port (int, optional): The port number for the connection. Defaults to 5672.
            vhost (str, optional): The virtual host for the connection. Defaults to "/".
            uid (str, optional): The user ID for the connection. Defaults to "".
            pwd (str, optional): The password for the connection. Defaults to "".
            metrics_url (str, optional): The URL for metrics. Defaults to "".
            metrics_type (str, optional): The type of metrics. Defaults to "".
            heartbeat (int, optional): The heartbeat interval in seconds. Defaults to 900.
            blocked_connection_timeout (int, optional): The timeout for blocked connections in seconds. Defaults to 600.
        Returns:
            None
        """

        self._pipeline = pipeline
        self._host = host
        self._port = port
        self._vhost = vhost
        self._uid = uid
        self._pwd = pwd
        self._metrics_url = metrics_url
        self._metrics_type = metrics_type
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._output_key = output_key
        self._output_type = output_type
        self._heartbeat = heartbeat
        self._blocked_connection_timeout = blocked_connection_timeout
        self._imagedir = imagedir
        self._result_connection: Optional[BlockingConnection] = None

        self._node_name = os.environ.get("NODE_NAME", "")
        self._pod_name = os.environ.get("POD_NAME", "")
        self._pod_namespace = os.environ.get("POD_NAMESPACE", "")
        self._pod_ip = os.environ.get("POD_IP", "")

        self._image_cache = ImageCache(imagedir)
        self._image_cache._init_cache()

    def _run_request_queue(self) -> None:
        """
        Continuously runs the request queue, connecting to the request service and consuming messages.
        This method establishes a connection to the request service and processes messages from the request queue.
        It handles message acknowledgment on success and negative acknowledgment on failure, allowing for message
        requeueing and eventual dropping if the requeue limit is reached. In case of connection errors, it attempts
        to reconnect after a short delay.
        Raises:
            AMQPChannelError: If there is an error with the AMQP channel.
            AMQPConnectionError: If there is an error with the AMQP connection.
        """

        while True:
            try:
                self._connect_to_request()
                logger.info(f"servicing request queue {self._request_queue}")

                # consume messages from the request queue, blocking for a maximum number of
                # seconds before returning to process heartbeats etc.
                while True:
                    for method_frame, properties, body in self._input_channel.consume(
                        self._request_queue,
                        inactivity_timeout=5,
                    ):
                        if method_frame:
                            # ack on success, nack on failure - the message will be requeued
                            # on nack and eventually dropped if the requeue limit is reached
                            try:
                                self._process_queue_input(
                                    self._input_channel, method_frame, properties, body
                                )
                                self._input_channel.basic_ack(method_frame.delivery_tag)
                            except Exception as e:
                                logger.exception(e)
                                self._input_channel.basic_nack(
                                    method_frame.delivery_tag
                                )

            except (AMQPChannelError, AMQPConnectionError):
                logger.warning("request connection closed, reconnecting")
                if self._input_channel and not self._input_channel.connection.is_closed:
                    logger.info("closing request connection")
                    self._input_channel.connection.close()
                sleep(5)

    def start_request_queue(self) -> None:
        """Start the request queue."""
        Thread(target=self._run_request_queue).start()

    def _connect_to_request(self) -> None:
        """
        Setup the connection, channel and queue to service incoming requests.
        """
        logger.info("connecting to request queue")
        if self._uid != "":
            credentials = pika.PlainCredentials(self._uid, self._pwd)
            self._request_connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    self._host,
                    self._port,
                    self._vhost,
                    credentials,
                    heartbeat=self._heartbeat,
                    blocked_connection_timeout=self._blocked_connection_timeout,
                )
            )
        else:
            self._request_connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    self._host,
                    heartbeat=self._heartbeat,
                    blocked_connection_timeout=self._blocked_connection_timeout,
                )
            )
        self._input_channel = self._request_connection.channel()
        # use quorum queues with a delivery limit to prevent infinite requeueing of a bad map
        self._input_channel.queue_declare(
            queue=self._request_queue,
            durable=True,
            arguments={"x-delivery-limit": REQUEUE_LIMIT, "x-queue-type": "quorum"},
        )
        self._input_channel.basic_qos(prefetch_count=1)

    def _get_connection_parameters(self) -> pika.ConnectionParameters:
        """
        Generates and returns the connection parameters for connecting to a RabbitMQ server.
        If a user ID (`_uid`) is provided, it includes the credentials in the connection parameters.
        Returns:
            pika.ConnectionParameters: The connection parameters for the RabbitMQ server.
        """
        if self._uid != "":
            credentials = pika.PlainCredentials(self._uid, self._pwd)
            return pika.ConnectionParameters(
                self._host,
                self._port,
                self._vhost,
                credentials,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=self._blocked_connection_timeout,
            )

        return pika.ConnectionParameters(
            self._host,
            heartbeat=self._heartbeat,
            blocked_connection_timeout=self._blocked_connection_timeout,
        )

    def _connect_to_result(self) -> None:
        """
        Setup the connection, channel and queue to service outgoing results.
        """
        logger.info("connecting to result queue")
        connectionParameters = self._get_connection_parameters()
        self._result_connection = pika.BlockingConnection(connectionParameters)

        if self._result_connection is not None:
            self._output_channel = self._result_connection.channel()
            self._output_channel.queue_declare(
                queue=self._result_queue,
                durable=True,
                arguments={"x-delivery-limit": REQUEUE_LIMIT, "x-queue-type": "quorum"},
            )

    def start_result_queue(self) -> None:
        """Start the result publishing thread."""
        Thread(target=self._run_result_queue).start()

    def _run_result_queue(self) -> None:
        """
        Main loop to service the result queue. process_data_events is set to block for a maximum
        of 1 second before returning to ensure that heartbeats etc. are processed.
        """
        while True:
            try:
                if self._result_connection is None or self._result_connection.is_closed:
                    logger.info(f"connecting to result queue {self._result_queue}")
                    self._connect_to_result()

                if self._result_connection is not None:
                    self._result_connection.process_data_events(time_limit=1)
                else:
                    logger.error("result connection not initialized")
            except (AMQPChannelError, AMQPConnectionError):
                logger.warn("result connection closed, reconnecting")
                if (
                    self._result_connection is not None
                    and self._result_connection.is_open
                ):
                    self._result_connection.close()
                sleep(5)

    def _publish_result(self, result: RequestResult) -> None:
        """
        Publish the result of a request to the output queue.

        Args:
            result: The result to publish.
        """
        if self._result_connection is not None:
            self._result_connection.add_callback_threadsafe(
                lambda: self._output_channel.basic_publish(
                    exchange="",
                    routing_key=self._result_queue,
                    body=json.dumps(result.model_dump()),
                )
            )
        else:
            logger.error("result connection not initialized")

    def _process_queue_input(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        props: spec.BasicProperties,
        body: bytes,
    ) -> None:
        """
        Process a request from the input queue.

        Args:
            channel: The channel the request was received on.
            method: The method used to deliver the request.
            body: The body of the request.

        """

        logger.info("request received from input queue")

        gauge_labels = {"labels": [{"name": "pod_name", "value": self._pod_name}]}

        try:
            body_decoded = json.loads(body.decode())
            # parse body as request
            request = Request.model_validate(body_decoded)

            # create the input
            image, image_path = self._get_image(request.image_id, request.image_url)
            input = self._create_pipeline_input(request, image)

            # add metric of job starting
            if self._metrics_url != "":
                requests.post(
                    self._metrics_url
                    + "/counter/"
                    + self._metrics_type
                    + "_started?step=1"
                )
                requests.post(
                    self._metrics_url
                    + "/gauge/"
                    + self._metrics_type
                    + "_working?value=1",
                    json=gauge_labels,
                )

            job_started_time = time.perf_counter()

            # run the pipeline
            outputs = self._pipeline.run(input)

            run_elasped_time = time.perf_counter() - job_started_time

            # create the response
            output_raw = outputs[self._output_key]
            if isinstance(output_raw, BaseModelOutput):
                result = self._create_output(request, str(image_path), output_raw)
            elif isinstance(output_raw, Output):
                result = self._create_empty_output(request, str(image_path), output_raw)
            else:
                raise ValueError("Unsupported output type")
            logger.info("writing request result to output queue")

            output_elasped_time = time.perf_counter() - run_elasped_time

            # run queue operations
            self._publish_result(result)

            publish_elasped_time = time.perf_counter() - output_elasped_time

            logger.info("result written to output queue")
            if self._metrics_url != "":
                requests.post(
                    self._metrics_url
                    + "/counter/"
                    + self._metrics_type
                    + "_completed?step=1"
                )
                requests.post(
                    self._metrics_url
                    + "/histogram/"
                    + self._metrics_type
                    + "_run?value="
                    + str(run_elasped_time)
                )
                requests.post(
                    self._metrics_url
                    + "/histogram/"
                    + self._metrics_type
                    + "_output?value="
                    + str(output_elasped_time)
                )
                requests.post(
                    self._metrics_url
                    + "/histogram/"
                    + self._metrics_type
                    + "_publish?value="
                    + str(publish_elasped_time)
                )
                requests.post(
                    self._metrics_url
                    + "/gauge/"
                    + self._metrics_type
                    + "_working?value=0",
                    json=gauge_labels,
                )
        except Exception as e:
            logger.exception(e)
            if self._metrics_url != "":
                requests.post(
                    self._metrics_url
                    + "/counter/"
                    + self._metrics_type
                    + "_errored?step=1"
                )
                requests.post(
                    self._metrics_url
                    + "/gauge/"
                    + self._metrics_type
                    + "_working?value=0",
                    json=gauge_labels,
                )

    def _create_pipeline_input(
        self, request: Request, image: PILImage
    ) -> PipelineInput:
        """
        Create the pipeline input for the request.

        Args:
            request: The request.
            image: The image.

        Returns:
            The pipeline input.
        """
        input = PipelineInput()
        input.image = image
        input.raster_id = request.image_id

        return input

    def _create_output(
        self, request: Request, image_path: str, output: BaseModelOutput
    ) -> RequestResult:
        """
        Create the output for the request.

        Args:
            request: The request.
            image_path: The path to the image.
            output: The output of the pipeline.

        Returns:
            The request result.
        """
        return RequestResult(
            request=request,
            output=json.dumps(output.data.model_dump()),
            success=True,
            image_path=image_path,
            output_type=self._output_type,
        )

    def _create_empty_output(
        self, request: Request, image_path: str, output: Output
    ) -> RequestResult:
        """
        Create the output for the request.

        Args:
            request: The request.
            image_path: The path to the image.
            output: The output of the pipeline.

        Returns:
            The request result.
        """
        return RequestResult(
            request=request,
            output="{}",
            success=True,
            image_path=image_path,
            output_type=self._output_type,
        )

    def _get_image(self, image_id: str, image_url: str) -> Tuple[PILImage, str]:
        """
        Retrieve an image either from the cache or by downloading it from a given URL.
        Args:
            image_id (str): The unique identifier for the image.
            image_url (str): The URL from which to download the image if it is not cached.
        Returns:
            Tuple[PILImage, str]: A tuple containing the image object and the path to the cached image.
        Raises:
            ValueError: If the image cannot be downloaded from the provided URL.
        """
        # check cache for the iamge
        image_path = self._image_cache._get_cache_doc_path(image_id)
        image = self._image_cache.fetch_cached_result(f"{image_id}.tif")
        if not image:
            # not cached - download from s3 and cache - we assume no credentials are needed
            # on the download url
            logger.info(f"cache miss - downloading image from {image_url}")
            image_file_reader = ImageFileReader()
            image = image_file_reader.process(image_url, anonymous=True)
            if not image:
                logger.error(f"failed to download image from {image_url}")
                raise ValueError("Failed to download image")
            self._image_cache.write_result_to_cache(image, f"{image_id}.tif")
        return (image, image_path)
