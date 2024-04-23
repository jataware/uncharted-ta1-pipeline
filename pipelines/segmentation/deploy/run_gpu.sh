#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --runtime=nvidia \
    --gpus all \
    --pull always \
    --rm \
    --name segmentation \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-segmentation:latest \
        --workdir /workdir \
        --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201 \
        --cdr_schema
