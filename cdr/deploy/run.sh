#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
# args: $2 - mode "host" or "process"
# args: $3 - ID of COG if in process mode

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --rm \
    --name cdr \
    -e CDR_API_TOKEN=$CDR_API_TOKEN \
    -e NGROK_AUTHTOKEN=$NGROK_AUTHTOKEN \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-cdr:latest --workdir $1 --mode host
