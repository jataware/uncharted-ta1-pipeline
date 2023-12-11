#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/point_extraction
cp ../*.py pipelines/point_extraction
cp ../pyproject.toml pipelines/point_extraction

cp -r ../../../schema .

cp -r ../../../tasks .

# run the build
docker build -t docker.uncharted.software/point_extraction:latest .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema

