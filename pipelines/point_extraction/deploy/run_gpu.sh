#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --runtime=nvidia \
    --gpus all \
    --rm \
    --name point_extraction \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-point-extract:latest \
    --imagedir /imagedir \
    --workdir /workdir \
    --model_point_extractor pipelines/point_extraction_weights/points.pt \
    --model_segmenter pipelines/segmentation_weights \
    --batch_size 20 \
    --rabbit_host rabbitmq
