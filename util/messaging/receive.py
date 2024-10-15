#!/usr/bin/env python
import json
import pprint
import pika, sys, os
import argparse
from tasks.common.request_client import RequestResult


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_name", help="Name of the queue", type=str)
    args = parser.parse_args()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()

    channel.queue_declare(
        queue=args.queue_name,
        durable=True,
        arguments={"x-queue-type": "quorum", "x-delivery-limit": 3},
    )

    def callback(ch, method, properties, body):
        body_decoded = json.loads(body.decode())
        request_result = RequestResult.model_validate(body_decoded)
        print(request_result)
        obj_output = json.loads(request_result.output)
        pprint.pprint(obj_output)

    channel.basic_consume(
        queue=args.queue_name, on_message_callback=callback, auto_ack=True
    )
    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
