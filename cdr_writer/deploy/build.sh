#!/bin/bash

# copy the files to the build directory
mkdir -p cdr_writer
cp ../*.py cdr_writer
cp ../pyproject.toml cdr_writer
cp -r ../../schema .
cp -r ../../tasks .
cp -r ../../util .

# run the build with the platform argument if provided, otherwise build for the host architecture
platform=${1:-}
if [[ -n "$platform" ]]; then
    echo "Platform: $platform"
    docker buildx build --platform "$platform" -t uncharted/lara-cdr-writer:latest . --load
else
    docker build -t uncharted/lara-cdr-writer:latest .
fi

# cleanup the temp files
rm -rf cdr_writer
rm -rf schema
rm -rf tasks
rm -rf util