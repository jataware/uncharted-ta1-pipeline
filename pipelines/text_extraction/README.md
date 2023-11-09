
## LARA Image Text Extraction (OCR) Pipeline 

This pipeline performs OCR-based text extraction on an image

This module currently uses Google-Vision OCR API by default:
https://cloud.google.com/vision/docs/ocr#vision_text_detection_gcs-python

See more info on pipeline tasks here: [../../tasks/text_extraction/README.md](../../tasks/text_extraction/README.md)

### Installation

* python=3.9 or higher is recommended
* The module is installed via the requirements.txt file
``` 
pip install -r requirements.txt
```
* Data i/o is done via REST (see below)
* To access the Google Vision API, the `GOOGLE_APPLICATION_CREDENTIALS` environment variable must be set to the google vision credentials json file (see example `.env` file)
* Input is an image (ie binary image file buffer)
* Ouput is OCR results as a list of JSON objects --  `PageExtraction` objects (from the CMA TA1 schema)

### REST API
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the text-extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint  

### Dockerized deployment
Work-in-progress

