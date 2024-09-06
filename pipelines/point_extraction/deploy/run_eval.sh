#!/bin/bash

# args: $1 - path to local directory to mount as /input in docker container
#       $2 - path to local directory to mount as /output in docker container
#       $3 - path to local directory to mount as /hints in docker container
#       $4 - path to local directory to mount as /workdir in docker container

# ensure that an the GOOGLE_APPLICATION_CREDENTIALS env var is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS env var must be set to the path of the GCP service account key"
    exit 1
fi

# ensure that the args are present
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: $0 <input_dir> <output_dir> <hints_dir> <workdir_dir>"
    exit 1
fi

docker network ls | grep -q 'lara' || docker network create lara
docker run \
    --pull always \
    --rm \
    --name point_extraction \
    --entrypoint python3 \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/input \
    -v $2:/output \
    -v $3:/hints \
    -v $4:/workdir \
    --net lara \
    uncharted/lara-point-extract:eval \
         -m pipelines.point_extraction.run_pipeline \
        --input /input \
        --output /output \
        --workdir /workdir \
        --legend_hints /hints \
        --bitmasks \
        --model_point_extractor pipelines/point_extraction_weights/points.pt \
        --model_segmenter pipelines/segmentation_weights