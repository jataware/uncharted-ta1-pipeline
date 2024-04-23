from enum import Enum
import json
import logging
from pathlib import Path
import pprint

import pika

from PIL.Image import Image as PILImage

from tasks.common.pipeline import (
    BaseModelOutput,
    Pipeline,
    PipelineInput,
)
from tasks.common.io import ImageFileInputIterator, download_file
from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import spec
from pydantic import BaseModel
from typing import Tuple

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

        self.setup_queues()

    def setup_queues(self) -> None:
        """
        Setup the input and output queues.
        """

        logger.info("wiring up request queue to input and output queues")

        # setup input and output queue
        request_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=900,
                blocked_connection_timeout=600,
            )
        )
        self._input_channel = request_connection.channel()
        self._input_channel.queue_declare(queue=self._request_queue)
        self._input_channel.basic_qos(prefetch_count=1)
        self._input_channel.basic_consume(
            queue=self._request_queue,
            on_message_callback=self._process_queue_input,
            auto_ack=True,
        )

        result_connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                self._host,
                heartbeat=self._heartbeat,
                blocked_connection_timeout=self._blocked_connection_timeout,
            )
        )
        self._output_channel = result_connection.channel()
        self._output_channel.queue_declare(queue=self._result_queue)

    def start_request_queue(self):
        """Start the request queue."""
        logger.info("starting request queue")
        self._input_channel.start_consuming()

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
            pprint.pprint(output_raw)
            if type(output_raw) == BaseModelOutput:
                result = self._create_output(request, str(image_path), output_raw)
            else:
                raise ValueError("Unsupported output type")
            logger.info("writing request result to output queue")

            # run queue operations
            self._output_channel.basic_publish(
                exchange="",
                routing_key=self._result_queue,
                body=json.dumps(result.model_dump()),
            )
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

        if not filename.exists():
            logger.info(f"image not found - downloading to {filename}")

            # download image
            image_data = download_file(image_url)

            # write it to working dir, creating the directory if necessary
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "wb") as file:
                file.write(image_data)

        # load images from file
        return filename, ImageFileInputIterator(str(filename))
