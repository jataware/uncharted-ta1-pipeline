#!/bin/bash

# copy the files to the build directory
mkdir -p cdr
cp ../*.py cdr
cp ../pyproject.toml cdr
cp -r ../../schema .
cp -r ../../tasks .

# run the build
docker build -t uncharted/lara-cdr:latest .

# cleanup the temp files
rm -rf cdr
rm -rf schema
rm -rf tasks
