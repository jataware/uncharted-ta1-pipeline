from enum import auto
from time import sleep
import pika

from tasks.common.queue import Request
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_name", type=str, help="Name of the queue")
    args = parser.parse_args()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    channel.queue_declare(
        queue=args.queue_name,
        durable=True,
        arguments={"x-queue-type": "quorum", "x-delivery-limit": 3},
    )

    for i in range(5):
        request = Request(
            id=str(i),
            task=str(i),
            image_id="c0089400862e476e4f66863283bb934d974d9b4d9a060d164b1e0b3685a6a127",
            image_url="https://s3.amazonaws.com/public.cdr.land/cogs/c0089400862e476e4f66863283bb934d974d9b4d9a060d164b1e0b3685a6a127.cog.tif",
            output_format="5",
        )

        channel.basic_publish(
            exchange="", routing_key=args.queue_name, body=request.model_dump_json()
        )
        sleep(1)
    connection.close()


if __name__ == "__main__":
    main()
