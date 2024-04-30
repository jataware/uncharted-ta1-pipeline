
## LARA Point Extraction Pipeline


This pipeline extracts bedding point symbols from a map, along with their orientation and associated incline information. The model leverages [YOLOv8](https://github.com/ultralytics/ultralytics) for the baseline object detection task.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Extracted Symbols

Initial efforts have focused on identifying and extracting the following symbols:
* Inclined Bedding (aka strike/dip)
* Vertical Bedding
* Horizontal Bedding
* Overturned Bedding
* Inclined Foliation
* Vertical Foliation
* Vertical Joint
* Sink Hole
* Gravel Borrow Pit
* Mine Shaft
* Prospect
* Mine Tunnel
* Mine Quarry

### Point Symbol Orientation
Some point symbols also contain directional information.
Point orientation (ie "strike" direction) and the "dip" magnitude are also extracted for applicable symbol types:
* Inclined Bedding (strike/dip)


### Installation

python 3.10 or higher is required

To install from the current directory:
```
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
* Ouput is the set of extracted points materialized as a:
  * `MapImage` JSON object, which contains a list of `MapPointLabel` capturing the point information
  * List of `FeatureResults` JSON objects as defined in the CMA TA1 CDR schema

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the point extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json

python3 -m pipelines.point_extraction.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir \
    --model_point_extractor https://s3/compatible/endpoint/point_extractor_model_weights_dir \
    --model_segmenter https://s3/compatible/endpoint/segmentation_model_weights_dir \
    --cdr_schema False \
    --bitmasks False \
    --legend_hints_dir /legend/hints/dir
```

Note that the `model_point_extractor` and `model_segmenter` parameters can point to a folder in the local file system, or to a resource on an S3-compatible endpoint.  The first file with a `.pt` extension will be loaded as the model weights.

In the S3 case, the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables must be set accordingly.  The model weights be fetched from the S3 endpoint and cached.

The `model_segmenter` param is optional. If present each image will be segmented and only the map area will be used for point symbol extraction.

OCR text extraction is used to extract the "dip" magnitude labels for strike/dip point symbols. For this functionality, the `GOOGLE_APPLICATION_CREDENTIALS` environment variables must be set.

If `cdr_schema` == True, the pipeline will also produce point extractions in the CDR JSON format.

If `bitmasks` = True, the pipeline will produce points extractions in bitmask image format (AI4CMA Contest format)

Legend hints JSON files (ie from the AI4CMA Contest) can be used to improve points extractions, using the `legend_hints_dir` parameter.


### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the segmenter pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export AWS_ACCESS_KEY_ID=<KEY ID>
export AWS_SECRET_ACCESS_KEY=<SECRET KEY>
export GOOGLE_APPLICATION_CREDENTIALS=/credentials/google_api_credentials.json

python3 -m pipelines.point_extraction.run_server \
    --workdir /model/workingdir \
    --model_point_extractor https://s3/compatible/endpoint/point_extractor_model_weights_dir \
    --model_segmenter https://s3/compatible/endpoint/segmentation_model_weights_dir
```

To test the server:

```
curl localhost:5000/healthcheck
```

To extract points from an image:

```
curl http://localhost:5000/api/process_image \
  --header 'Content-Type: image/tiff' \
  --data-binary @/path/to/map/image.tif
  --output output/file/path.json
```

### Dockerized deployment

A dockerized version REST service described above is available that includes model weights.  Similar to the steps above,
OCR text extraction for "dip" magnitude labels need the `GOOGLE_APPLICATION_CREDENTIALS` environment variables must be set, and
a local directory for caching results must be provided.  To run:

```
cd deploy
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/google_api_credentials.json
./run.sh /path/to/workdir
```

The `deploy/build.sh` script can also be used to build the Docker image from source.


