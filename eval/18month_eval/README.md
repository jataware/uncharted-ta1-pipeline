## LARA Self-Evaluation Code for the CriticalMAAS Program 18-month Evaluation 

### Ground Truth Data

* The expected ground truth data for the self-evaluation is from the `data.tar.gz` package prepared by Mitre for the 18-month program evaluation. This includes ground truth data for both georeferencing and feature extraction.


### Georeferencing Evaluation

#### 1. Prepare groumd truth data
* Run the `georef_groundtruth_prepare.py` script
* __Note:__ the ENV variables for input data paths need to be set __before hand__
* This will load the geoTIFFs and ground truth GCPs, and for each map the following will be saved in the `results/georef/` folder:
    - input raster TIFF (pixel-space)
    - AI4CMA contest-style query pts as a CSV (in both NAD83 coords and pixels)

#### 2. Run LARA georeferencing and get metrics
* Use the output from step 1, and process the TIFF rasters using the LARA georeferencing pipeline. More info is [here](../../pipelines/geo_referencing/README.md).
* Specifically, use LARA's georeferencing `run_pipeline.py` module with:
    - `--input` set to the TIFF rasters from step 1
    - `--query_dir` set to the CSV query files from step 1
    - `--diagnostics` enabled to get a summary CSV file of RMSE results per map


### Points Feature Extraction Evaluation

#### 1. Prepare groumd truth data
* Run the `points_extraction_groundtruth_prepare.py` script
* __Note:__ the ENV variables for input data paths need to be set __before hand__
* This will load the geoTIFFs, ground truth GCPs, and shapefiles.
* Point-based Shapefile features will be parsed and converted to pixel-space (using the geoTIFF GCP-based transform)
* For each map the following will be saved in the `results/feature_extraction/` folder:
    - input raster TIFF (pixel-space)
    - Pixel coordinates of point-based features in JSON format

#### 2. Inspect ground truth feature labels
* Manually inspect the ground truth features from step 1 and associate each feature per map with the LARA [16-type point ontology](../../pipelines/point_extraction/README.md), where possible.
* This step is necessary because NGMDB feature shapefiles don't necessary use a common naming convention.
* For any point features that are in the ontology, re-name the feature-based JSON files from step 1 using the following convention: `<raster COG ID>_...__<point ontology label>.json`

#### 3. Run LARA points extraction
* Use the output from step 1, and process the TIFF rasters using the LARA points extraction pipeline. More info is [here](../../pipelines/point_extraction/README.md).
* Specifically, use LARA's point extraction `run_pipeline.py` module with:
    - `--input` set to the TIFF rasters from step 1
    - `--cdr_schema` enabled to get JSON results in CDR format

#### 4. Calculate point extraction metrics
* Run the `points_extraction_calc_metrics.py` script
* __Note:__ the ENV variables for input data paths (from steps 1 and 3) need to be set __before hand__
* Results for F1 scores per feature per map will be save to a summary CSV file in the `results/feature_extraction/` folder