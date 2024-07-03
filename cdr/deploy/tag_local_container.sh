#!/bin/bash

docker tag uncharted/lara-cdr:$1 uncharted/lara-cdr:$2
docker tag uncharted/lara-georef:$1 uncharted/lara-georef:$2
docker tag uncharted/lara-point-extract:$1 uncharted/lara-point-extract:$2
docker tag uncharted/lara-segmentation:$1 uncharted/lara-segmentation:$2
docker tag uncharted/lara-metadata-extract:$1 uncharted/lara-metadata-extract:$2

docker push uncharted/lara-cdr:$2
docker push uncharted/lara-georef:$2
docker push uncharted/lara-point-extract:$2
docker push uncharted/lara-segmentation:$2
docker push uncharted/lara-metadata-extract:$2

