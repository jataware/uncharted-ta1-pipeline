## LARA Tasks
LARA Pipeline Tasks


### Installation

* python=3.9 or higher is recommended
* The task library is pip install-able from this directory:
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
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python


#### Running Text Extraction Task

* Text extraction is done via the `TextExtractor` child classes
* To access the Google Vision API, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the google vision credentials json file
* Input is a map raster as an OpenCV image
* Ouput is OCR results as a `DocTextExtraction` object

### Map Metadata Extraction Task

**Goal:** to extract metadata values such as title, author and scale from text

This module uses the OpenAI interface to incorporate GPT output: https://platform.openai.com/

#### Running Metadata Extraction

* Metadata extraction is done throught the `MetadataExtraction` class
* A valid OpenAPI key must be supplied through the `OPENAI_API_KEY` environment variable
* Input is a `DocTextExtraction` object containing previously extracted map text
* Output is a `MetadataExtraction` object containing the identified map metadata

### Image Segmentation Task

**Goal:** to perform segmentation to isolate the map and legend regions on an image

Segmentation is done using a fine-tuned version of the `LayoutLMv3` model:
https://github.com/microsoft/unilm/tree/master/layoutlmv3

See more info on pipeline deployment here: [../../pipelines/segmentation/README.md](../../pipelines/segmentation/README.md)

#### Segmentation categories (classes)

The model currently supports 3 segmentation classes:
* Map
* Legend (polygons)
* Legend (points and lines)

#### Running Map Segementation Tasks
* Model inference is controlled via the `DetectronSegmenter` class
* Input is a map raster (as an OpenCV image)
* Ouput is segmentation polygon results as a `MapSegmentation` object

#### GPU Inference

TBD -- These deployments currently support CPU model inference only