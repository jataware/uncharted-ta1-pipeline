## LARA Tasks
LARA Pipeline Tasks


### Installation

* python=3.9 or higher is recommended
* The module is pip install-able from this directory:
```
pip install -e .
```

The image segmentation task relies on [LayoutLMV3](https://github.com/microsoft/unilm/tree/master/layoutlmv3), which is not `pip` installable due to transitive dependencies on specific verions of Pytorch and Detectron2.  Use of the image segmentation consequently requires manual installation of both libraries separately.

For GPU support (requires CUDA version >= 11.1):

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

For software only:

```bash
pip install torch==1.10.0 torchvision==0.11.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

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