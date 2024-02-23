docker network create lara
docker run \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
    -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/workdir \
    --net lara \
    -p 5000:5000 \
    docker.uncharted.software/metadata-extraction:latest --workdir /workdir --model $2
