# Uncharted TA1 Models

![example workflow](https://github.com/uncharted-lara/lara-models/actions/workflows/build_test.yml/badge.svg)

## LARA - Layered Atlas Reconstruction Analytics
This repository contains Uncharted's TA1 contributions for DARPA's CriticalMAAS program. The main goals are automated feature extraction and georeferencing of geologic maps.

This repository contains five pipelines:

* [Map Segmentation](pipelines/segmentation/README.md) - detects and extracts the main map area, polygon legend, point/line legend and geologic cross sections from maps
* [Metadata Extraction](pipelines/metadata_extraction/README.md) - extracts metadata values such as title, author, year and scale from an input map image
* [Point Extraction](pipelines/point_extraction/README.md) - detects and extracts the location and orientation of geologic point symbols from an input map image
* [Georeferencing](pipelines/geo_referencing/README.md) - computes an image space to geo space transform given an input map image
* [Text Extraction](pipelines/text_extraction/README.md) - extracts text as individual words, lines or paragraphs/blocks from an input image

The `tasks` directory contains the `pip` installable library of tasks and supporting utilities, with each pipeline found in the `pipelines` directory being composed of these tasks.  Each pipeline is itself `pip` installable, and is accompanied by a wrapper to support command line execution (`run_pipeline.py`), and a server wrapper to support execution as a REST service (`run_sever.py`).  Scripts to build the server wrapper into a Docker container are also included.



