#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --pull always \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    -p 5000:5000 \
    uncharted/lara-text-extract:latest \
        --workdir /workdir \
        --cdr_schema
