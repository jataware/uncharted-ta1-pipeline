
## LARA Image Text Extraction (OCR) Pipeline

This pipeline performs OCR-based text extraction on an image

This module currently uses Google-Vision OCR API by default:
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python

See more info on pipeline tasks here: [../../tasks/text_extraction/README.md](../../tasks/README.md)

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
* Output is the set of extracted text items materialized as a:
  * `DocTextExtraction` JSON object
  * List of `PageExtraction` objects as defined in the CMA TA1
  * [Geopackage](geopackage.org) adhering to the CMA TA1 schema

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the map extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentials/google_api_credentials.json

python3 -m pipelines.text_extraction.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir \
    --tile True \
    --pixel_limit 1024
```

Where `tile` inidicates whether the image should be tiled or resized when it is larger than `pixel_limit`.



### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the metadata extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json

python3 -m pipelines.metadata_extraction.run_server \
    --workdir /model/workingdir
    --tile True \
    --pixel_limit 1024
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be started as a container:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/credentials/google_api_credentials.json

./run.sh /model/working/dir
```



