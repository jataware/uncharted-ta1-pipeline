import argparse
import json
import logging
import os

import pika

from PIL.Image import Image as PILImage

from pipelines.geo_referencing.factory import create_geo_referencing_pipeline
from pipelines.geo_referencing.output import LARAModelOutput, JSONWriter
from tasks.common.pipeline import (
    BaseModelOutput,
    ObjectOutput,
    OutputCreator,
    Pipeline,
    PipelineInput,
)
from tasks.common.io import ImageFileInputIterator
from util.image import download_file

from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import spec

from pydantic import BaseModel

from typing import Any, Optional, Tuple

LARA_REQUEST_QUEUE_NAME = "lara-request"
LARA_RESULT_QUEUE_NAME = "lara-result"

logger = logging.getLogger("process_queue")


class Request(BaseModel):
    id: str
    task: str
    image_id: str
    image_url: str
    output_format: str


class RequestResult(BaseModel):
    request: Request

    success: bool
    output: str
    image_path: str


class RequestQueue:

    def __init__(self, working_dir: str):
        self._working_dir = working_dir
        logger.info(f"initialize queue using work dir {working_dir}")

    def setup_queue(self, input: Tuple[Channel, str], output: Tuple[Channel, str]):
        logger.info("wiring up request queue to input and output queues")

        # write the results to the output channel
        self._output_channel = output
        self._input_channel = input

    def run_queue(self):
        logger.info("starting request queue")
        # read requests from the input channel
        self._input_channel[0].basic_qos(prefetch_count=1)
        self._input_channel[0].basic_consume(
            queue=self._input_channel[1], on_message_callback=self._process_queue_input
        )
        self._input_channel[0].start_consuming()

    def _process_queue_input(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        properties: spec.BasicProperties,
        body: bytes,
    ):
        logger.info("request received from input queue")
        body_decoded = json.loads(body.decode())
        # parse body as request
        request = Request.model_validate(body_decoded)
        channel.basic_ack(delivery_tag=method.delivery_tag)

        # process the request
        result = self._process_request(request)
        logger.info("writing request result to output queue")

        # run queue operations
        self._output_channel[0].basic_publish(
            exchange="",
            routing_key=self._output_channel[1],
            body=json.dumps(result.model_dump()),
        )
        logger.info("result written to output queue")

    def _process_request(self, request: Request) -> RequestResult:
        image_path, image_it = self._get_image(
            self._working_dir, request.image_id, request.image_url
        )
        # get the right pipeline
        pipeline = self._get_pipeline(request)

        # create the input
        input = self._create_pipeline_input(request, next(image_it)[1])

        # run the pipeline
        outputs = pipeline.run(input)

        # create the response
        output_raw: BaseModelOutput = outputs["lara"]  # type: ignore
        return self._create_output(request, image_path, output_raw)

    def _get_output(self, request: Request) -> OutputCreator:
        match request.output_format:
            case "cdr":
                return LARAModelOutput("lara")
        raise Exception("unrecognized output format specified in request")

    def _get_pipeline(self, request: Request) -> Pipeline:
        output = self._get_output(request)
        # TODO: USE THE REQUEST TO FIGURE OUT THE PROPER PIPELINE TO CREATE
        return create_geo_referencing_pipeline(
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_xsection_20231201",
            [output],
        )

    def _create_pipeline_input(
        self, request: Request, image: PILImage
    ) -> PipelineInput:
        input = PipelineInput()
        input.image = image
        input.raster_id = request.image_id

        return input

    def _create_output(
        self, request: Request, image_path: str, output: BaseModelOutput
    ) -> RequestResult:
        return RequestResult(
            request=request,
            output=json.dumps(output.data.model_dump()),
            success=True,
            image_path=image_path,
        )

    def _get_image(
        self, working_dir: str, image_id: str, image_url: str
    ) -> Tuple[str, ImageFileInputIterator]:
        # check working dir for the image
        disk_filename = os.path.join(working_dir, f"{image_id}.tif")

        if not os.path.isfile(disk_filename):
            # download image
            image_data = download_file(image_url)

            # write it to working dir
            with open(disk_filename, "wb") as file:
                file.write(image_data)

        # load image from disk
        return disk_filename, ImageFileInputIterator(disk_filename)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--request_queue", type=str, default="localhost")
    parser.add_argument("--result_queue", type=str, default="localhost")
    p = parser.parse_args()

    logger.info(
        f"starting queue process listening for requests at {p.request_queue} and writing results to {p.result_queue}"
    )
    os.makedirs(os.path.dirname(p.workdir), exist_ok=True)

    # setup input and output queue
    request_connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            p.request_queue, heartbeat=900, blocked_connection_timeout=600
        )
    )
    request_channel = request_connection.channel()
    request_channel.queue_declare(queue=LARA_REQUEST_QUEUE_NAME)

    result_connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            p.result_queue, heartbeat=900, blocked_connection_timeout=600
        )
    )
    result_channel = result_connection.channel()
    result_channel.queue_declare(queue=LARA_RESULT_QUEUE_NAME)

    # start the queue
    queue = RequestQueue(p.workdir)
    queue.setup_queue(
        (request_channel, LARA_REQUEST_QUEUE_NAME),
        (result_channel, LARA_RESULT_QUEUE_NAME),
    )

    queue.run_queue()


if __name__ == "__main__":
    main()
