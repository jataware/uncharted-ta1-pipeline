
## LARA Image Segmentation Tasks

**Goal:** to perform segmentation to isolate the map and legend regions on an image

Segmentation is done using a fine-tuned version of the `LayoutLMv3` model: 
https://github.com/microsoft/unilm/tree/master/layoutlmv3

See more info on pipeline deployment here: [../../pipelines/segmentation/README.md](../../pipelines/segmentation/README.md)

### Segmentation categories (classes)

The model currently supports 3 segmentation classes:
* Map
* Legend (polygons)
* Legend (points and lines)

### Installation

* python=3.9 or higher is recommended
* The module is pip install-able from this directory:
``` 
pip install -e .
```
* Model inference is controlled via the `DetectronSegmenter` class
* Input is a map image (ie as numpy array)
* Ouput is segmentation polygon results as a JSON object, either as a `SegmentationResults` object or a `PageExtraction` object (the latter being part of the CMA TA1 schema) 

### GPU Inference

TBD -- These deployments currently support CPU model inference only



