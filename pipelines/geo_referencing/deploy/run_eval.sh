#!/bin/bash

# args: $1 - path to local directory to mount as /input in docker container
#       $2 - path to local directory to mount as /output in docker container
#       $3 - path to local directory to mount as /query in docker container
#       $4 - path to local directory to mount as /hints in docker container
#       $5 - path to local directory to mount as /workdir in docker container

# ensure that an the GOOGLE_APPLICATION_CREDENTIALS env var is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS env var must be set to the path of the GCP service account key"
    exit 1
fi

# ensure that an the OPENAI_API_KEY env var is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY env var must be set to the OpenAI API key"
    exit 1
fi

# ensure that the args are present
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
    echo "Usage: $0 <input_dir> <output_dir> <points_dir> <hints_dir> <workdir>"
    exit 1
fi

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --pull always \
    --name georeferencing \
    --entrypoint python3 \
    --rm \
    -it \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/input \
    -v $2:/output \
    -v $3:/query \
    -v $4:/hints \
    -v $5:/workdir \
    --net lara \
    uncharted/lara-georef:dry-run \
        -m pipelines.geo_referencing.run_pipeline \
        --input /input \
        --output /output \
        --points_dir /input \
        --query_dir /query \
        --clue_dir /hints \
        --workdir /workdir \
        --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201
