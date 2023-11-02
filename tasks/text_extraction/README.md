
## LARA Image Text Extraction Tasks

**Goal:** to perform OCR-based text extraction on an image

This module currently uses Google-Vision OCR API by default:
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python


### Installation

* python=3.9 or higher is recommended
* The module is pip install-able from this directory:
``` 
pip install -e .
```

### Running Text Extraction Tasks

* Text extraction is done via the `TextExtractor` child classes
* To access the Google Vision API, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the google vision credentials json file 
* Input is a map image 
* Ouput is OCR results as a list of JSON objects --  `PageExtraction` objects (from the CMA TA1 schema)






