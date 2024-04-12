#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --runtime=nvidia \
    --gpus all \
    --pull always \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-metadata-extract:latest \
        --workdir /workdir \
        --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201 \
        --cdr_schema
