#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
#       $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --runtime=nvidia \
    --gpus all \
    --pull always \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-metadata-extract:latest \
        --workdir /workdir \
        --imagedir /imagedir \
        --model pipelines/segmentation_weights/layoutlmv3_20240531 \
        --cdr_schema \
        --rabbit_host rabbitmq
