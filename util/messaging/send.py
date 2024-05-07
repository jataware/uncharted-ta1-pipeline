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

    for i in range(1):
        request = Request(
            id="000001",
            task="2",
            image_id="test_image",
            image_url="https://s3.amazonaws.com/public.cdr.land/cogs/c0089400862e476e4f66863283bb934d974d9b4d9a060d164b1e0b3685a6a127.cog.tif",
            output_format="5",
        )

        channel.basic_publish(
            exchange="", routing_key=args.queue_name, body=request.model_dump_json()
        )
        print(f"Sent {request.model_dump_json()}")
    connection.close()


if __name__ == "__main__":
    main()
