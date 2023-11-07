
## LARA Image Segmentation Pipeline


This pipeline performs segmentation to isolate the map and legend regions on an image

Segmentation is done using a fine-tuned version of the `LayoutLMv3` model: 
https://github.com/microsoft/unilm/tree/master/layoutlmv3

See more info on pipeline tasks here: [../../tasks/segmentation/README.md](../../tasks/segmentation/README.md)

### Segmentation categories (classes)

The model currently supports 3 segmentation classes:
* Map
* Legend (polygons)
* Legend (points and lines)

### Installation

* python=3.9 or higher is recommended
* The module is installed via the requirements.txt file
``` 
pip install -r requirements.txt
```
* Data i/o is done via REST (see below)
* Model weights can be input from S3 or local drive
* Input is an image (ie binary image file buffer)
* Ouput is segmentation polygon results as a JSON object, either as a `SegmentationResults` object or a `PageExtraction` object (the latter being part of the CMA TA1 schema)

`inference_example.py` is an example script to process a single image

### REST API
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the segmenter pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint  

### Dockerized deployment
* `docker_build.sh` script can be used to build the pipeline `lara/segmenter` docker image. (Note: is a two-stage build, with the `tasks` module used as a base image)
* `docker-compose.yml` is a sample configuration for the legend-and-map segmentation pipeline

### GPU Inference

TBD -- These deployments currently support CPU model inference only
