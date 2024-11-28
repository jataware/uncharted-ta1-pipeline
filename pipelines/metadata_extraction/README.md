
## LARA Metadata Extraction Pipeline


This pipeline extracts metadata such as title, year and scale from an input raster map image.  Extracted text is
combined with a request for the fields of interest into a prompt, which is passed to an [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o#gpt-4o) for analysis and extraction.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Extracted Fields

The following are the currently extracted fields along with an example of each:

| Field | Example |
|-------|---------|
| Title |BEDROCK GEOLOGIC MAP SHOWING THICKNESS OF OVERLYING QUATERNARY DEPOSITS, GRAND ISLAND QUADRANGLE, NEBRASKA AND KANSAS |
| Authors | Howlett, J.A., Xavier, C.F., Pryde, K.A. |
| Year | 1971 |
| Scale | 1:24000 |
| Quadrangle | Mount Union |
| Coordinate Systems | Arizona Coordinate System, Central Zone |
| Base Map String | U.S. Geological Survey, 1961 |
| Projection | Polyconic |
| Datum | NAD 1927 |
| Coordinate Reference System (CRS) | EPSG:4267 |
| Counties | Mariposa |
| States | US-AZ |
| Country | US |
| Places | Tin Cup Mine, Hickory Butte |
| Population Centers | Davidson, Dalton's Corners |
| Language | English |

### Installation

* python 3.10 or higher is required
* Installation of Detectron2 requires `torch` already be present in the environment, so it must be installed manually.
* Note: for python virtual environments, `conda` is more reliable for installing torch==2.0.x than `venv`

To install from the current directory:
```
# manually install torch - this is necessary due to issues with detectron2 dependencies
# (see https://github.com/facebookresearch/detectron2/issues/4472)
pip install torch==2.0.1

# install the task library
cd ../../tasks
pip install -e .[segmentation]

# install the metadata extraction pipeline
cd ../pipelines/metadata_extraction
pip install -e .
```

The segmentation dependencies are required due to the use of the map segmentation task within the metadata extraction pipeline.

*Depending on the shell used, the brackets may need to be escaped.*

### Overview ###

* Pipeline is defined in `metdata_extraction_pipeline.py` and is suitable for integration into other systems
* Input is a image (ie binary image file buffer)
* Output is the set of metadata fields materialized as:
  * `MetadataExtraction` JSON object (LARA's internal data schema) and/or
  * `CogMetaData` JSON object (part of the CDR TA1 schema)

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the metadata extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.metadata_extraction.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model /path/to/segmentation/model/weights \
    --llm gpt-4o (gpt model string - see OpenAI model page for options) \
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --no_gpu (if set, pipeline will force CPU-only processing)
```

Note that when the segmentation `model` parameter can point to a folder in the local file system, or to a resource on an S3-compatible endpoint. The folder/resource must contain the following files:
* `config.yaml`
* `config.json`
* `model_final.pth`

In the S3 case, the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables must be set accordingly.  The model weights an configuration files will be fetched from the S3 endpoint and cached.

### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the metadata extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

The server can also be configured to run with a request queue, using RabbitMQ, if the `rest` flag is not set.

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.metadata_extraction.run_server \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model /path/to/segmentation/model/weights \
    --llm gpt-4o (which gpt model version to use) \
    --rest (if set, run the server in REST mode, instead of resquest-queue mode)
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --no_gpu (if set, pipeline will force CPU-only processing) \
    --imagedir /pipeline/images/working/dir (only needed for request-queue mode) \
    --rabbit_host (rabbitmq host; only needed for request-queue mode)
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be started as a container:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/credentials/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

./run.sh /pipeline/working/dir /pipeline/image/dir
```

Alternatively, a [Makefile](../../Makefile) is available to handle the building and deploying the various LARA pipeline containers.

