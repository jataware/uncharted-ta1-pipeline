#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/geo_referencing
cp ../*.py pipelines/geo_referencing
cp ../pyproject.toml pipelines/geo_referencing

cp -r ../../../schema .

cp -r ../../../tasks .

cp -r ../../../util .

# run the build
docker build -t docker.uncharted.software/geo-ref:latest .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util

