#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/geo_referencing
cp ../*.py pipelines/geo_referencing

cp -r ../../../tasks .

# run the build
docker build -t docker.uncharted.software/georef:0.1 .

# cleanup the temp files
rm -rf pipelines/geo_referencing
rm -rf tasks