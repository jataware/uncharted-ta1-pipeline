#!/bin/bash

# if $1 is not set, then set it to the default value
if [ -z "$1" ]
then
    GPU=cpu
else
    GPU=gpu
fi

# copy the files to the build directory
mkdir -p pipelines/segmentation
cp ../*.py pipelines/segmentation
cp ../pyproject.toml pipelines/segmentation

cp -r ../../../schema .

cp -r ../../../tasks .

# run the build
echo $GPU
docker build --build-arg GPU=$GPU -t docker.uncharted.software/segmentation:latest .

# cleanup the temp files
rm -rf pipelines/segmentation
rm -rf tasks
rm -rf schema

