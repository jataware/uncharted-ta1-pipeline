#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/geo_referencing
cp ../*.py pipelines/geo_referencing

cp -r ../../../schema .

cp -r ../../../compute .

cp -r ../../../model .

cp -r ../../../util .

cp -r ../../../tasks .

# run the build
docker build --no-cache -t docker.uncharted.software/georef:0.1 .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf compute
rm -rf model
rm -rf util