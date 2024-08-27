#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/metadata_extraction
cp ../*.py pipelines/metadata_extraction
cp ../pyproject.toml pipelines/metadata_extraction

cp -r ../../../schema .
cp -r ../../../util .
cp -r ../../../tasks .


mkdir -p pipelines/segmentation_weights
if [ -z "$1" ]
then
    echo "ERROR - No segment model weights dir supplied"
    segment_model=""
    exit 1
else
    segment_model=$1
    echo "Segment model weights dir: $segment_model"
    cp -r $segment_model/* pipelines/segmentation_weights
fi

# run the build with the platform argument if provided, otherwise build for the host architecture
platform=${2:-}
if [[ -n "$platform" ]]; then
    echo "Platform: $platform"
    docker buildx build --platform "$platform" -t uncharted/lara-metadata-extract:latest . --load
else
    docker build -t uncharted/lara-metadata-extract:latest .
fi

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util
