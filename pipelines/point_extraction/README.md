
## LARA Point Extraction Pipeline


This pipeline extracts bedding point symbols from a map, along with their orientation and associated incline information. The model leverages [YOLOv8](https://github.com/ultralytics/ultralytics) for the baseline object detection task.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Extracted Symbols

Initial efforts have focused on identifying and extracting the following symbols:
* Inclined Bedding (aka strike/dip)
* Vertical Bedding
* Horizontal Bedding
* Overturned Bedding

### Point Symbol Orientation
Some point symbols also contain directional information.
Point orientation (ie "strike" direction) and the "dip" magnitude are also extracted for applicable symbol types:
* Inclined Bedding (strike/dip)

### Installation

python=3.9 or higher is recommended

To install from the current directory:
```
# install the task library
cd ../../tasks
pip install -e .[point]

# install the point extraction pipeline
cd ../pipelines/point_extraction
pip install -e .
```

### Overview ###

* Pipeline is defined in `point_extraction_pipeline.py` and is suitable for integration into other systems
* Input is a image (ie binary image file buffer)
* Ouput is extracted points as a `MapImage` JSON object, which contains a list of `MapPointLabel` caputring the point information.

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the point extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.point_extraction.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir \
    --model_point_extractor https://s3/compatible/endpoint/point_extractor_model_weights_dir \
    --model_segmenter https://s3/compatible/endpoint/segmentation_model_weights_dir 
```

Note that the `model_point_extractor` and `model_segmenter` parameters can point to a folder in the local file system, or to a resource on an S3-compatible endpoint.  The first file with a `.pt` extension will be loaded as the model weights.

In the S3 case, the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables must be set accordingly.  The model weights be fetched from the S3 endpoint and cached.

The `model_segmenter` param is optional. If present each image will be segmented and only the map area will be used for point symbol extraction.

### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the segmenter pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.point_extraction.run_server \
    --workdir /model/workingdir \
    --model_point_extractor https://s3/compatible/endpoint/point_extractor_model_weights_dir \
    --model_segmenter https://s3/compatible/endpoint/segmentation_model_weights_dir 
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be started as a container:

```
cd deploy

export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

./run.sh /model/workingdir https://s3/compatible/endpoint/model_weights_dir
```


