#!/bin/bash

# copy the files to the build directory
mkdir -p pipelines/text_extraction
cp ../*.py pipelines/text_extraction
cp ../pyproject.toml pipelines/text_extraction

cp -r ../../../schema .
cp -r ../../../util .
cp -r ../../../tasks .

# run the build
docker build -t uncharted/lara-text-extract:latest .

# cleanup the temp files
rm -rf pipelines
rm -rf tasks
rm -rf schema
rm -rf util

