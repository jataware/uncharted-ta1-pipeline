
## LARA Metadata Extraction Pipeline


This pipeline extracts metadata such as title, year and scale from an input raster map image.  Extracted text is
combined with a request for the fields of interest into a prompt, which is passed to an [OpenAI GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)
for analysis and extraction.

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
| Vertical Datum | Vertical Datum of 1929 |
| Coordinate Systems | Arizona Coordinate System, Central Zone |
| Base Map String | U.S. Geological Survey, 1961 |
| Projection | Polyconic |
| Datum | NAD 1927 |
| Counties | Mariposa |
| States | Arizona |
| Country | United States |

### Installation

python 3.10 or higher required

To install from the current directory:
```
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
* Output is the set of metadata fields materialized as a:
  * `MetadataExtraction` JSON object
  * `Map` JSON object adhering to the CMA TA1 schema)
  * [Geopackage](geopackage.org) adhering to the CMA TA1 schema

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the map extraction pipeline, and allows for a directory map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.metadata_extraction.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir \
    --model /segmentation/model/weights
```

The `model` argument should point to the latest segmentation model weights, which can be stored on the local file system, or accessed through a URL.

### REST Service ###
`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the metadata extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.metadata_extraction.run_server \
    --workdir /model/workingdir
```

### Dockerized deployment
The `deploy/build.sh` script can be used to build the server above into a Docker image.  Once built, the server can be started as a container:

```
cd deploy

export GOOGLE_APPLICATION_CREDENTIALS=/credentials/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

./run.sh /model/working/dir
```


