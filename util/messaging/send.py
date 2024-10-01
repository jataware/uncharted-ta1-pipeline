from ast import arg
import json
from time import sleep
import pika

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_name", type=str, help="Name of the queue")
    parser.add_argument("--cog_id", type=str, help="ID of the COG")
    parser.add_argument("--input", type=str, help="Input file")
    args = parser.parse_args()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    channel.queue_declare(
        queue=args.queue_name,
        durable=True,
        arguments={"x-queue-type": "quorum", "x-delivery-limit": 3},
    )

    # load the list of cogs from the input file if present into an array, otherwise add the
    # single cog specified by the cog_id argument to the array
    cogs = []
    if args.input:
        with open(args.input, "r") as f:
            cogs = f.readlines()
    elif arg:
        cogs.append(args.cog_id)

    #    a request for each cog in the array
    for i, cog in enumerate(cogs):
        request = {
            "id": str(i),
            "task": str(i),
            "image_id": cog,
            "image_url": f"https://s3.amazonaws.com/public.cdr.land/cogs/{cog}.cog.tif",
            "output_format": "5",
        }

        channel.basic_publish(
            exchange="", routing_key=args.queue_name, body=json.dumps(request)
        )
        sleep(1)
    connection.close()


if __name__ == "__main__":
    main()
