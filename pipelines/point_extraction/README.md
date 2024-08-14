
## LARA Point Extraction Pipeline


This pipeline extracts bedding point symbols from a map, along with their orientation and associated incline (dip) information. The model leverages [YOLO](https://github.com/ultralytics/ultralytics) for extraction of high priority / common point symbol types. In addition, a CV-based One-Shot algorithm can be used to extract less common point symbols.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Extracted Symbols

#### Object Detection Model
The YOLO object detection model has been trained to extract common geologic point symbols as follows:
* Inclined Bedding (aka strike/dip)
* Vertical Bedding
* Horizontal Bedding
* Overturned Bedding
* Inclined Foliation
* Inclined Foliation (Igneous)
* Vertical Foliation
* Vertical Joint
* Sink Hole
* Lineation
* Drill Hole
* Gravel Borrow Pit
* Mine Shaft
* Prospect
* Mine Tunnel
* Mine Quarry

#### One-Shot Model
The One-shot CV algorithm can be used to extract any leftover less common point symbols that may be present. This algorithm requires legend swatches to be available as a template (either via HITL annotation or some other manner)

### Point Symbol Orientation
Many point symbols also contain directional information.
Point orientation (ie "strike" direction) and the "dip" magnitude are also extracted for applicable symbol types:
* Inclined Bedding (strike/dip)
* Vertical Bedding
* Overturned Bedding
* Inclined Foliation
* Inclined Foliation (Igneous)
* Vertical Foliation
* Vertical Joint
* Lineation
* Mine Tunnel


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
pip install -e .[point,segmentation]

# install the point extraction pipeline
cd ../pipelines/point_extraction
pip install -e .
```

The segmentation dependencies are required due to the use of the map segmentation task within the metadata extraction pipeline.

*Depending on the shell used, the brackets may need to be escaped.*

### Overview ###

* Pipeline is defined in `point_extraction_pipeline.py` and is suitable for integration into other systems
* Input is a image (ie binary image file buffer)
* Ouput is the set of extracted points materialized as:
  * `PointLabels` JSON object (LARA's internal data schema) and/or
  * `FeatureResults` JSON object (part of the CDR TA1 schema)

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the point extraction pipeline, and allows for a directory of map images to be processed serially.

To run from the repository root directory:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json

python3 -m pipelines.point_extraction.run_pipeline \
    --input /image/input/dir \
    --output /results/output/dir \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model_point_extractor /path/to/points/yolo/model/weights \
    --model_segmenter /path/to/segmentation/model/weights \
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --fetch_legend_items (if set, the pipeline will query the CDR for validated legend annotations for a given map input) \
    --no_gpu (if set, pipeline will force CPU-only processing) \
    --bitmasks (if set, pipeline will also output legacy CMAAS contest-style bitmasks) \
    --legend_hints_dir /input/legend/hints/dir  (to input legacy CMAAS contest legend hints)
```

Note that the `model_point_extractor` and `model_segmenter` parameters can point to a folder in the local file system, or to a resource on an S3-compatible endpoint.  The first file with a `.pt` extension will be loaded as the model weights.

In the S3 case, the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables must be set accordingly.  The model weights be fetched from the S3 endpoint and cached.

OCR text extraction is used to extract the "dip" magnitude labels for strike/dip point symbols. For this functionality, the `GOOGLE_APPLICATION_CREDENTIALS` environment variables must be set.

If `cdr_schema` is set, the pipeline will also produce point extractions in the CDR JSON format.

The `fetch_legend_items` option requires the `CDR_API_TOKEN` environment variable to be set. The pipeline will query the CDR to check for validated legend annotations for a given map input. This is optional. It is not needed for the main YOLO model, but required for the secondary One-Shot algorithm.

#### Legacy AI4CMA Contest Processing

* If `bitmasks` is set, the pipeline will produce points extractions in bitmask image format (AI4CMA Contest format)
* Legend hints JSON files (ie from the AI4CMA Contest) can be used to improve points extractions, using the `legend_hints_dir` parameter.


### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the segmenter pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

The server can also be configured to run with a request queue, using RabbitMQ, if the `rest` flag is not set.

To start the server:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>
export GOOGLE_APPLICATION_CREDENTIALS=/path/to//google_api_credentials.json
export CDR_API_TOKEN=<SECRET TOKEN>

python3 -m pipelines.point_extraction.run_server \
    --workdir /pipeline/working/dir (default is tmp/lara/workdir) \
    --model_point_extractor /path/to/points/yolo/model/weights \
    --model_segmenter /path/to/segmentation/model/weights \
    --rest (if set, run the server in REST mode, instead of resquest-queue mode) \
    --cdr_schema (if set, pipeline will also output CDR schema JSON objects) \
    --fetch_legend_items (if set, the pipeline will query the CDR for validated legend annotations for a given map input) \
    --no_gpu (if set, pipeline will force CPU-only processing) \
    --imagedir /pipline/images/working/dir (only needed for request-queue mode) \
    --rabbit_host (rabbitmq host; only needed for request-queue mode) 
```

### Dockerized deployment

A dockerized version REST service described above is available that includes model weights.  Similar to the steps above,
OCR text extraction for "dip" magnitude labels need the `GOOGLE_APPLICATION_CREDENTIALS` environment variables must be set, and a local directory for caching results must be provided.  To run:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
export CDR_API_TOKEN=<SECRET TOKEN>

./run.sh /path/to/workdir /pipline/images/working/dir
```

The `deploy/build.sh` script can also be used to build the Docker image from source.


