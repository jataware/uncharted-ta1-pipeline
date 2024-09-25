#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
#       $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --pull always \
    --rm \
    --name segmentation \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-segmentation:test \
        --workdir /workdir \
        --imagedir /imagedir \
        --model pipelines/segmentation_weights \
        --cdr_schema \
        --rabbit_host rabbitmq
