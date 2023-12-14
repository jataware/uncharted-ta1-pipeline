
## LARA Georeferencing Pipeline


This pipeline georeferences an input raster map image. The georeferencing approach revolves around parsing text in the image to extract coordinates, and then using those coordinates to georeference ground control points. It currently attempts to parse Degrees, Minutes, Seconds style coordinates and UTM style coordinates.

If run through CLI, the georeferencing system is currently structured to execute 5 independent pipelines, each producing 5 outputs.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Pipelines
As it stands, there are 5 pipelines that are executed for each map. The pipelines follow the same general process, with slight variations as not all maps are created equal. The differences are in how the OCR process handles larger images and how the Region Of Interest is obtained.

** Resize **
The Resize pipeline will resize images to fit under the maximum allowable size limit for the OCR processing. The Region Of Interest is determined via an entropy based approach.

** Tile **
The Tile pipeline will tile the image using the minimum amount of tiles necessary to have them all fit under the allowable size limit for the OCR processing. The Region Of Interest is determined via an entropy based approach.

** Fixed ROI **
The Tile pipeline will tile the image using the minimum amount of tiles necessary to have them all fit under the allowable size limit for the OCR processing. The Region Of Interest is determined by the image segmentation model, with the map area being buffered by a fixed amount of pixels (150 for now)

** Image ROI **
The Tile pipeline will tile the image using the minimum amount of tiles necessary to have them all fit under the allowable size limit for the OCR processing. The Region Of Interest is determined by the image segmentation model, with the map area being buffered by a percentage of the overall image size (3% for now)

** ROI ROI **
The Tile pipeline will tile the image using the minimum amount of tiles necessary to have them all fit under the allowable size limit for the OCR processing. The Region Of Interest is determined by the image segmentation model, with the map area being buffered by a percentage of the ROI size (5% for now)

### Output

5 different outputs are produced when georeferencing is executed at the command line.

** GCP  List **
A list of georeferenced GCPs along with distance from ground truth and other error indications if available. The data is output as a CSV file with the filename being `test-{pipeline name}.csv`.

** Summary **
A list of maps with their RMSE if available output as a CSV file with the name `test_summary-{pipeline name}.csv`.

** Schema **
A list of projections of maps containing the input GCPs georeferenced. The data is output following the specified TA1 schema in a JSON file named `test_schema-{pipeline name}.json`.

** Levers **
All data that can be manipulated by the user to control or influence the georeferencing of maps. These can be the raw text and locations of parsed coordinates or the region of interest (map area). Future potential levers include extracted metadata, derived geofence, or hemisphere. The file is named `test_levers-{pipeline name}.json`.

** GCPs **
A list of georeferenced gcps output specifically for downstream integration containing only the bare minimum information for user validation. The file is named `test_gcps-{pipeline name}.json`.

### Installation

python=3.9 or higher is recommended

To install from the current directory:
```
# install the task library
cd ../../tasks
pip install -e .

# install the metadata extraction pipeline
cd ../pipelines/geo_referencing
pip install -e .
```

### Overview ###

* Pipelines are defined in `factory.py` and cannot be composed into other pipelines
* Input is a image (ie binary image file buffer)
* Output are files output to disk

### Command Line Execution ###
`run_pipeline.py` provides a command line wrapper around the georeferencing pipelines, allowing for a directory of map images to be processed serially.

To run from the repository root directory:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.geo_referencing.run_pipeline \
    --input /image/input/dir \
    --output /model/output/dir \
    --workdir /model/working/dir
```

### REST Service ###
The REST service runs the Fixed ROI pipeline only to produce a single schema formatted output.

`run_server.py` provides the pipeline as a REST service with the following endpoints:
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the georeferencing pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

To start the server:
```
export GOOGLE_APPLICATION_CREDENTIALS=/credentinals/google_api_credentials.json
export OPENAI_API_KEY=<OPEN API KEY>

python3 -m pipelines.geo_referencing.run_server \
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


