#!/bin/bash

# args $1 - path to point model weights
#      $2 - path to segmentation model weights

# validate the args
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <point_model_weights_path> <segmentation_model_weights_path>"
    exit 1
fi

./build.sh
pwd
cd ../../pipelines/point_extraction/deploy
./build.sh $1 $2
cd ../../pipelines/segmentation/deploy
./build.sh $2
cd ../../pipelines/geo_referencing/deploy
./build.sh $2
cd ../../pipelines/metadata_extraction/deploy
./build.sh $2
