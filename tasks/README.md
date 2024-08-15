## LARA Tasks
LARA Pipeline Tasks
* In general, a `task` is considered a processing step in a given LARA `pipeline`.
* All `tasks` are executed via their `run` method 


### Installation

* python 3.10 or higher is required
* The task library is pip installable from this directory:
```
pip install -e .
```

The point detection and segmentation tasks both have extra dependencies that are quite extensive, so those are managed as a optional requirements.

To install each run:

```
pip install -e .[segmentation]
pip install -e .[point]
```

*Depending on the shell used, the brackets may need to be escaped.*

### Image Text Extraction (OCR) Task

**Goal:** to perform OCR-based text extraction on an image

This module currently uses Google-Vision OCR API by default:
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python.


#### Using the Text Extraction Task

* Text extraction is done via the `TextExtractor` child classes
* To access the Google Vision API, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the google vision credentials json file
* Input is a map raster image
* Ouput is OCR results as a `DocTextExtraction` object

A pipeline using this task, along with a CLI and sever wrapper are available at [../pipelines/text_extraction](../pipelines/text_extraction)

### Map Metadata Extraction Task

**Goal:** to extract metadata values such as title, author and scale from text

This module uses the OpenAI interface to incorporate GPT output: https://platform.openai.com/

#### Using the Metadata Extraction Task

* Metadata extraction is done through the `MetadataExtraction` class
* A valid OpenAPI key must be supplied through the `OPENAI_API_KEY` environment variable
* Input is a `DocTextExtraction` object containing previously extracted map text (liklely from a `TextExtractor` task)
* Output is a `MetadataExtraction` object containing the identified map metadata

A pipeline using this task, along with a CLI and sever wrapper are available at [../pipelines/metadata_extraction](../pipelines/metadata_extraction)

### Image Segmentation Task

**Goal:** to perform segmentation to isolate the map, legend and cross-section regions on an image

Segmentation is done using a fine-tuned version of the `LayoutLMv3` model:
https://github.com/microsoft/unilm/tree/master/layoutlmv3

See more info on pipeline deployment here: [../pipelines/segmentation](../pipelines/segmentation)

#### Segmentation categories (classes)

The model currently supports 3 segmentation classes:
* Map
* Legend (polygons)
* Legend (points and lines)
* Cross sections

#### Using the Map Segementation Task
* Model inference is controlled via the `DetectronSegmenter` class
* Input is a map raster (as an OpenCV image)
* Ouput is segmentation polygon results as a `MapSegmentation` object

A pipeline using this task, along with a CLI and sever wrapper are available at [../pipelines/segmentation](../pipelines/segmentation)

### Point Extraction Tasks ###

**Goal:** Extracts point symbols from a map, along with their orientation and associated incline (dip) information

The model leverages [YOLO](https://github.com/ultralytics/ultralytics) for extracting high priority / common point symbol types. In addition, a CV-based One-Shot algorithm can be used to extract less common point symbols.

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
The One-shot CV algorithm can be used to extract any "leftover" less common point symbols that may be present. This algorithm requires legend swatches to be available as a template (either via human-in-the-loop annotation or some other manner)

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


#### Using the Point Extraction Tasks ####
* The main YOLO point extraction is available in the `YOLOPointDetector` task
* The One-Shot point extraction is in the `TemplateMatchPointExtractor` task
* Point orientation (strike/dip angle) detection is in the `PointOrientationExtractor` task
* Output is a`PointLabels` JSON object, which contains a list of `PointLabel` capturing the point information.
* Extraction tasks take `ImageTiles` objects as inputs - `ImageTiles` are produced by the `Tiler` task
* `ImageTiles` can be re-assembled into a `PointLabels` using the `Untiler` task

A pipeline using these task, along with a CLI and sever wrapper are available at [../pipelines/point_extraction](../pipelines/point_extraction)

### Georeferencing Tasks ###

**Goal:** map the pixel coordinates to world coordinates including extraction of the projection

This module relies on image segmentation to identify the map area, OCR output as basis for georeferencing, and metadata extraction to build a geofence and to identify potential geocoding opportunities.

#### Using the Georeferencing Task

* Georeferencing is done through a series of tasks culminating with the `GeoReference` class
* A valid OpenAPI key must be supplied through the `OPENAI_API_KEY` environment variable
* To access the Google Vision API, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the google vision credentials json file
* The path to the segmentation weights file must be provided as argument
* Output is a list of `QueryPoint` object containing the mapped coordinates and error if ground truth is specified, as well as the extracted projection

A pipeline using this task, along with a CLI and sever wrapper are available at [../pipelines/geo_referencing](../pipelines/geo_referencing)

