import json
import logging
import threading
from time import sleep
from typing import List, Optional

from pika.adapters.blocking_connection import BlockingChannel as Channel
from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from pika.exceptions import AMQPChannelError, AMQPConnectionError
from tasks.common.request_client import Request, RequestResult


logger = logging.getLogger("request_publisher")


class LaraRequestPublisher:
    """
    Class to publish LARA requests to a RabbitMQ queue.
    """

    REQUEUE_LIMIT = 3
    INACTIVITY_TIMEOUT = 5
    HEARTBEAT_INTERVAL = 900
    BLOCKED_CONNECTION_TIMEOUT = 600

    def __init__(
        self,
        request_queues: List[str],
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
    ) -> None:
        self._request_connection: Optional[BlockingConnection] = None
        self._request_channel: Optional[Channel] = None
        self._host = host
        self._port = port
        self._vhost = vhost
        self._uid = uid
        self._pwd = pwd
        self._request_queues = request_queues
        self._stop_event = threading.Event()

    def start_lara_request_queue(self):
        """
        Starts the LARA request queue by running the `run_request_queue` function in a separate thread.

        Args:
            host (str): The host address to pass to the `run_request_queue` function.

        Returns:
            None
        """
        self._stop_event.clear()
        threading.Thread(
            target=self._run_request_queue,
        ).start()

    def stop_lara_request_queue(self):
        """
        Stops the LARA request queue by setting the stop event.

        Returns:
            None
        """
        logger.info("Stopping request queue thread")
        self._stop_event.set()

    def publish_lara_request(self, req: Request, request_queue: str):
        """
        Publishes a LARA request to a specified queue.

        Args:
            req (Request): The LARA request object to be published.
            request_channel (Channel): The channel used for publishing the request.
            request_queue (str): The name of the queue to publish the request to.
        """
        logger.info(f"sending request {req.id} for image {req.image_id} to lara queue")
        if self._request_connection is not None and self._request_channel is not None:
            self._request_connection.add_callback_threadsafe(
                lambda: self._request_channel.basic_publish(  #   type: ignore
                    exchange="",
                    routing_key=request_queue,
                    body=json.dumps(req.model_dump()),
                )
            )
            logger.info(f"request {req.id} published to {request_queue}")
        else:
            logger.error("request connection / channel not initialized")

    def _create_channel(self) -> Channel:
        """
        Creates a blocking connection and channel on the given host and declares the given queue.

        Args:
            host: The host to connect to.
            queue: The queue to declare.

        Returns:
            The created channel.
        """
        logger.info(f"creating channel on host {self._host}")
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
        for queue in self._request_queues:
            channel.queue_declare(
                queue=queue,
                durable=True,
                arguments={
                    "x-delivery-limit": self.REQUEUE_LIMIT,
                    "x-queue-type": "quorum",
                },
            )
        return channel

    def _run_request_queue(self):
        """
        Main loop to service the request queue. process_data_events is set to block for a maximum
        of 1 second before returning to ensure that heartbeats etc. are processed.
        """
        self._request_connection: Optional[BlockingConnection] = None
        while self._stop_event.is_set() is False:
            try:
                if (
                    self._request_connection is None
                    or self._request_connection.is_closed
                ):
                    logger.info(
                        f"connecting to request queue {','.join(self._request_queues)}"
                    )
                    self._request_channel = self._create_channel()
                    self._request_connection = self._request_channel.connection

                if self._request_connection is not None:
                    self._request_connection.process_data_events(time_limit=1)
                else:
                    logger.error("request connection not initialized")
            except (AMQPChannelError, AMQPConnectionError):
                logger.warn("request connection closed, reconnecting")
                if (
                    self._request_connection is not None
                    and self._request_connection.is_open
                ):
                    self._request_connection.close()
                sleep(5)

        logger.info("request queue thread stopped")
