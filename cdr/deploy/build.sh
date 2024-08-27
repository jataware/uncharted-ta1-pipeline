#!/bin/bash

# copy the files to the build directory
mkdir -p cdr
cp ../*.py cdr
cp ../pyproject.toml cdr
cp -r ../../schema .
cp -r ../../tasks .
cp -r ../../util .

# run the build with the platform argument if provided, otherwise build for the host architecture
platform=${1:-}
if [[ -n "$platform" ]]; then
    echo "Platform: $platform"
    docker buildx build --platform "$platform" -t uncharted/lara-cdr:latest .
else
    docker buildx build -t uncharted/lara-cdr:latest .
fi

# cleanup the temp files
rm -rf cdr
rm -rf schema
rm -rf tasks
rm -rf util