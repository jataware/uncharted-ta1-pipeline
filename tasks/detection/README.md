### LARA Point Object Detector Tasks

**GOAL:** Extract point features from a map.

**Supported Points:** Inclined, Horizontal, Vertical and Overturned Beddings

## Installation
* Python >= 3.10 is recommended
* Export necessary AWS environment variables before running tasks as a pipeline.
  * `export AWS_ACCESS_KEY_ID='your_key_here'
  * `export AWS_SECRET_ACCESS_KEY='your_secret_here'
  * `AWS_S3_ENDPOINT_URL=https://s3.t1.uncharted.software`
    
* Install required packages via setup.py
  * `cd` into `lara-models/tasks/detection` and run `pip install -e .`
    
## Running the pipeline 

* Running this tool on a machine with a GPU is recommended. CPU is also supported, but is much slower.
* The full point detection pipeline is run via the CLI tool `run_pipeline.py`
* Supply the CLI script with the following arguments:
  * `--input_path`, the path to an untiled, raw map in `.tif` format
  * `--ckpt_name`, a named checkpoint of a pretrained object detector located under `https://s3.t1.uncharted.software/lara/models/points/`
  * `--output_path`, the output location of the point detection JSON

_Example Useage_

```
python detection/run_pipeline.py --input_path path/to/GEO_0454.tif --ckpt yolov8.pt --output_path path/to/pipeline_output.json
```

## Schema Formatting

Updating the format of the model output should be controlled by modifying the `serialize()` methods of Pydantic data objects within `entities.py`
