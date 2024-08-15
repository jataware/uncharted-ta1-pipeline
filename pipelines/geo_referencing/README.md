
## LARA Georeferencing Pipeline


This pipeline georeferences an input raster map image. The georeferencing approach revolves around parsing text in the image to extract coordinates, and then using those coordinates to georeference ground control points. It currently attempts to parse Degrees, Minutes, Seconds style coordinates and UTM style coordinates. There is initial support to extract potential geocoding options as basis for extraction of latitude and longitude transformations.

The georeferencing pipeline uses info from many other LARA tasks to improve its results:
* Text extraction (OCR) is used to extract map geo-coord labels and other metadata
* Metadata Extraction is used to help determine additional info about a map's general region and scale (e.g. state, county, UTM zone)
* Segmentation is used to determine to the map's bounds within an image.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)


### CLI Output

Five different outputs are produced when georeferencing is executed at the command line.

**GCP List**
A list of georeferenced GCPs along with distance from ground truth and other error indications if available. The data is output as a CSV file with the filename being `test-{pipeline name}.csv`.

**Summary**
A list of maps with their RMSE if available output as a CSV file with the name `test_summary-{pipeline name}.csv`.

**Schema**
A list of projections of maps containing the input GCPs georeferenced. The data is output following the specified TA1 schema in a JSON file named `test_schema-{pipeline name}.json`.

**Levers**
All data that can be manipulated by the user to control or influence the georeferencing of maps. These can be the raw text and locations of parsed coordinates or the region of interest (map area). Future potential levers include extracted metadata, derived geofence, or hemisphere. The file is named `test_levers-{pipeline name}.json`.

**GCPs**
A list of georeferenced gcps output specifically for downstream integration containing only the bare minimum information for user validation. The file is named `test_gcps-{pipeline name}.json`.

### Installation

* python 3.10 or higher is required
* Installation of Detectron2 requires `torch` to already be present in the environment, so it must be installed manually.
* Note: for python virtual environments, `conda` is more reliable for installing torch==2.0.x than `venv`

To install from the current directory:
```
# manually install torch - this is necessary due to issues with detectron2 dependencies
# (see https://github.com/facebookresearch/detectron2/issues/4472)
pip install torch==2.0.1

# install the task library
cd ../../tasks
pip install -e .[segmentation]

# install the georeferencing pipeline
cd ../pipelines/geo_referencing
pip install -e .
```

### Overview ###

* The georeferencing pipeline is defined in `factory.py` and is suitable for integration into other systems
* Input is a image (ie binary image file buffer)
* Output is the georeferencing results materialized as:
  * `GeoreferenceResult` JSON object (part of the CDR TA1 schema)

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the georeferencing pipeline, allowing for a directory of map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.geo_referencing.run_pipeline \
    --input /image/input/dir \
    --output /results/output/dir \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model /path/to/segmentation/model/weights \
    --extract_metadata True \
    --no_gpu (if set, pipeline will force CPU-only processing)
```

* `extract_metadata` Indicates whether to use metadata extraction to aid with georeferencing.

### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the georeferencing pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

The server can also be configured to run with a request queue, using RabbitMQ, if the `rest` flag is not set.

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>

python3 -m pipelines.geo_referencing.run_server \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model /path/to/segmentation/model/weights \
    --rest (if set, run the server in REST mode, instead of resquest-queue mode)
    --no_gpu (if set, pipeline will force CPU-only processing) \
    --imagedir /pipeline/images/working/dir (only needed for request-queue mode) \
    --rabbit_host (rabbitmq host; only needed for request-queue mode) 
```


### Dockerized deployment

A dockerized version REST service described above is available that includes model weights. OCR text and metadata extraction require the `GOOGLE_APPLICATION_CREDENTIALS` and `OPENAI_API_KEY` environment variables be set, and a
a local directory for caching results must be provided.  To run:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

./run.sh /path/to/workdir /pipeline/images/working/dir
```

The `deploy/build.sh` script can also be used to build the Docker image from source.

Alternatively, a [Makefile](../../Makefile) is available to handle the building and deploying the various LARA pipeline containers. 