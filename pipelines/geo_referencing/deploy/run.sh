#!/bin/bash

# args: $1 - path to local directory to mount as /workdir in docker container
#       $2 - path to local directory to mount as /imagedir in docker container

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --pull always \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    -v $2:/imagedir \
    --net lara \
    -p 5000:5000 \
    uncharted/lara-georef:latest \
        --workdir /workdir \
        --imagedir /imagedir \
        --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201
