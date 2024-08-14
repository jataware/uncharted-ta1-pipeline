
## LARA Image Text Extraction (OCR) Pipeline

This pipeline performs OCR-based text extraction on an image

This module currently uses Google-Vision OCR API by default:
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Installation

python 3.10 or higher is required

To install from the current directory:
```
# install the task library
cd ../../tasks
pip install -e .

# install the text extraction pipeline
cd ../pipelines/text_extraction
pip install -e .
```

### Overview ###

* Pipeline is defined in `text_extraction_pipeline.py` and is suitable for integration into other systems
* Input is a image (ie binary image file buffer)
* Output is the set of extracted text items materialized as:
  * `DocTextExtraction` JSON object (LARA's internal data schema) and/or
  * `FeatureResults` JSON object (part of the CDR TA1 schema)

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the text extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json

python3 -m pipelines.text_extraction.run_pipeline \
    --input /image/input/dir \
    --output /results/output/dir \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --tile True \
    --pixel_limit 6000 \
    --gamma_corr 1.0
```

* `tile` Indicates whether the image should be tiled or resized when it is larger than `pixel_limit`.
* `gamma_corr` Controls optional image gamma correction as pre-processing. Must be <= 1.0; default value is 1.0 (ie gamma correction disabled)


### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the metadata extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

The server can also be configured to run with a request queue, using RabbitMQ, if the `rest` flag is not set.

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json

python3 -m pipelines.text_extraction.run_server \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --tile True \
    --pixel_limit 6000 \
    --gamma_corr 1.0 \
    --rest (if set, run the server in REST mode, instead of resquest-queue mode) \
    --imagedir /pipline/images/working/dir (only needed for request-queue mode) \
    --rabbit_host (rabbitmq host; only needed for request-queue mode) 
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be started as a container:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json

./run.sh /model/working/dir /pipline/images/working/dir
```



