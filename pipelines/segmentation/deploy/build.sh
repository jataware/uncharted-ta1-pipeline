#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/segmentation
cp ../*.py pipelines/segmentation
cp ../pyproject.toml pipelines/segmentation

cp -r ../../../schema .

cp -r ../../../tasks .

# run the build
echo $GPU
docker build -t docker.uncharted.software/segmentation:latest .

# cleanup the temp files
rm -rf pipelines/segmentation
rm -rf tasks
rm -rf schema

