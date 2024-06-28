#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/geo_referencing
cp ../*.py pipelines/geo_referencing
cp ../pyproject.toml pipelines/geo_referencing

cp -r ../../../schema .
cp -r ../../../tasks .
cp -r ../../../util .

mkdir -p pipelines/segmentation_weights
if [ -z "$1" ]
then
    echo "ERROR - No segment model weights dir supplied"
    segment_model=""
    exit 1
else
    segment_model=$1
    echo "Segment model weights dir: $segment_model"
    cp -r $segment_model pipelines/segmentation_weights
fi

# run the build
docker build -t uncharted/lara-georef:latest .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util

