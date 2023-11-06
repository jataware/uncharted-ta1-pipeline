docker run \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
    -v $1:/input \
    -v $2:/output \
    -v $3:/workdir \
    docker.uncharted.software/metadata-extraction:latest
