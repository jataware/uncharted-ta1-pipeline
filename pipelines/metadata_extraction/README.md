
## LARA Map Metadata Extraction Pipeline

This pipeline extracts map metdata such as title, scale, authors and year.

This module currently uses GPT-3.5 Turbo.

See more info on pipeline tasks here: [../../tasks/README.md](../../tasks/README.md)

### Installation

* python=3.9 or higher is recommended
* The module is installed via the requirements.txt file
```
pip install -e .
```
* Data i/o is done via REST (see below)
* Input is an image (ie binary image file buffer)

### REST API
* ```POST:  /api/process_image``` - Sends an image (as binary file buffer) to the text-extraction pipeline for analysis. Results are JSON string.
* ```GET /healthcheck``` - Healthcheck endpoint

### Dockerized deployment

