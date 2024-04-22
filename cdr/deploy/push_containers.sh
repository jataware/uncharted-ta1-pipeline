#!/bin/bash

docker tag uncharted/lara-cdr:latest uncharted/lara-cdr:$1
docker tag uncharted/lara-georef:latest uncharted/lara-georef:$1
docker tag uncharted/lara-point-extract:latest uncharted/lara-point-extract:$1
docker tag uncharted/lara-segmentation:latest uncharted/lara-segmentation:$1

docker push uncharted/lara-cdr:$1
docker push uncharted/lara-georef:$1
docker push uncharted/lara-point-extract:$1
docker push uncharted/lara-segmentation:$1
