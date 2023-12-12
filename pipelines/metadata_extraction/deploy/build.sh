#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/metadata_extraction
cp ../*.py pipelines/metadata_extraction
cp ../pyproject.toml pipelines/metadata_extraction

cp -r ../../../schema .

cp -r ../../../tasks .

# run the build
docker build -t docker.uncharted.software/metadata-extraction:latest .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema

