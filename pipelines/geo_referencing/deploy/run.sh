docker network create lara
docker run \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    docker.uncharted.software/geo-ref:latest --workdir /workdir --model pipelines/segmentation_weights/layoutlmv3_xsection_20231201
