#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --rm \
    --name cdr \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-cdr:latest --workdir /workdir