import pika

from tasks.common.queue import Request
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_name", type=str, help="Name of the queue")
    args = parser.parse_args()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    channel.queue_declare(queue=args.queue_name)

    request = Request(
        id="000001",
        task="2",
        image_id="test_image",
        image_url="https://ngmdb.usgs.gov/ngm-bin/pdp/download.pl?q=2867_21357_5",
        output_format="5",
    )

    channel.basic_publish(
        exchange="", routing_key=args.queue_name, body=request.model_dump_json()
    )
    print(f"Sent {request.model_dump_json()}")
    connection.close()


if __name__ == "__main__":
    main()
