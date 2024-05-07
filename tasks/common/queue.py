from concurrent.futures import thread
from enum import Enum
import json
import logging
import os
from pathlib import Path
from threading import Thread
from time import sleep

from cv2 import log
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from PIL.Image import Image as PILImage

from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineInput,
)
from tasks.common.io import ImageFileInputIterator, download_file
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, spec
from pydantic import BaseModel
from typing import Optional, Tuple

logger = logging.getLogger("process_queue")

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


class RequestQueue:
    """
    Input and output messages queues for process pipeline requests and publishing
    the results.

    Args:
        pipeline: The pipeline to use for processing requests.
        request_queue: The name of the request queue.
        result_queue: The name of the result queue.
        host: The host of the queue.
        heartbeat: The heartbeat interval.
        blocked_connection_timeout: The blocked connection timeout.
        workdir: Intermediate output storage directory.
        imagedir: Drectory for storing source images.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        request_queue: str,
        result_queue: str,
        output_key: str,
        output_type: OutputType,
        workdir: Path,
        imagedir: Path,
        host="localhost",
        heartbeat=900,
        blocked_connection_timeout=600,
    ) -> None:
        """
        Initialize the request queue.
        """
        self._pipeline = pipeline
        self._host = host
        self._request_queue = request_queue
        self._result_queue = result_queue
        self._output_key = output_key
        self._output_type = output_type
        self._heartbeat = heartbeat
        self._blocked_connection_timeout = blocked_connection_timeout
        self._working_dir = workdir
        self._imagedir = imagedir
        self._result_connection: Optional[BlockingConnection] = None

    def _run_request_queue(self):
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
                        auto_ack=True,
                    ):
                        if method_frame:
                            self._process_queue_input(
                                self._input_channel, method_frame, properties, body
                            )

            except (AMQPChannelError, AMQPConnectionError):
                logger.warn("request connection closed, reconnecting")
                if self._input_channel and not self._input_channel.connection.is_closed:
                    logger.info("closing request connection")
                    self._input_channel.connection.close()
                sleep(5)

    def start_request_queue(self):
        """Start the request queue."""
        Thread(target=self._run_request_queue).start()

    def _connect_to_request(self):
        """
        Setup the connection, channel and queue to service incoming requests.
        """
        logger.info("connecting to request queue")
        self._request_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=self._blocked_connection_timeout,
            )
        )
        self._input_channel = self._request_connection.channel()
        self._input_channel.queue_declare(queue=self._request_queue)
        self._input_channel.basic_qos(prefetch_count=1)

    def _connect_to_result(self):
        """
        Setup the connection, channel and queue to service outgoing results.
        """
        logger.info("connecting to result queue")
        self._result_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=self._blocked_connection_timeout,
            )
        )
        self._output_channel = self._result_connection.channel()
        self._output_channel.queue_declare(queue=self._result_queue)

    def start_result_queue(self):
        """Start the result publishing thread."""
        Thread(target=self._run_result_queue).start()

    def _run_result_queue(self):
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

        try:
            body_decoded = json.loads(body.decode())
            # parse body as request
            request = Request.model_validate(body_decoded)

            # create the input
            image_path, image_it = self._get_image(
                self._imagedir, request.image_id, request.image_url
            )
            input = self._create_pipeline_input(request, next(image_it)[1])

            # run the pipeline
            outputs = self._pipeline.run(input)

            # create the response
            output_raw = outputs[self._output_key]
            if type(output_raw) == BaseModelOutput:
                result = self._create_output(request, str(image_path), output_raw)
            else:
                raise ValueError("Unsupported output type")
            logger.info("writing request result to output queue")

            # run queue operations
            self._publish_result(result)
            logger.info("result written to output queue")
        except Exception as e:
            logger.exception(e)

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

    def _get_image(
        self, imagedir: Path, image_id: str, image_url: str
    ) -> Tuple[Path, ImageFileInputIterator]:
        """
        Get the image for the request.
        """
        # check working dir for the image
        filename = imagedir / f"{image_id}.tif"

        if not os.path.exists(filename):
            logger.info(f"image not found - downloading to {filename}")

            # download image
            image_data = download_file(image_url)

            # write it to working dir, creating the directory if necessary
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "wb") as file:
                file.write(image_data)

        # load images from file
        return filename, ImageFileInputIterator(str(filename))
