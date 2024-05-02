#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --runtime=nvidia \
    --gpus all \
    --pull always \
    --rm \
    --name segmentation \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-segmentation:latest \
        --workdir /workdir \
        --imagedir /imagedir \
        --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201 \
        --cdr_schema \
        --rabbit_host rabbitmq
