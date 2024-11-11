#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --rm \
    --name cdr_writer \
    -e CDR_API_TOKEN=$CDR_API_TOKEN \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-cdr-writer:latest --workdir $1 --imagedir $2
