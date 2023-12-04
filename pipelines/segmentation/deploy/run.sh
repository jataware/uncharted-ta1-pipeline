#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - s3 url pointing to model folder
docker network create lara
docker run \
    --rm \
    --name segmentation \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    docker.uncharted.software/segmentation:latest --workdir /workdir --model $2
