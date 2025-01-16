from PIL import Image
import os, json
import rasterio
from pathlib import Path
from pyproj import Transformer
from eval_utils import score_query_points

"""
Script to prepare georeferencing ground truth results for the 18-month self-evaluation
Input data is assumed to follow the directory structure in the 18-month eval data.tar.gz package from Mitre
See README.md for more info
"""

# ---- ENV variables to edit

INPUT_MAPS_DIR = "<local/path/to/georeferencing/eval/geotiffs>"

COG_ID_TO_RECORD_JSON = "<local/path/to/ground_truth/cog_id_info.json>"

# -------

RESULTS_DIR_GEOREF = "results/georef"
CRS_TARGET = "EPSG:4269"
Image.MAX_IMAGE_PIXELS = 500000000


def run():
    print(f"*** Running georef_groundtruth_prepare...")

    os.makedirs(RESULTS_DIR_GEOREF, exist_ok=True)

    record2cog = {}
    if COG_ID_TO_RECORD_JSON:
        with open(COG_ID_TO_RECORD_JSON, "r") as fp:
            record2cog = json.load(fp)

    summary_results_csv = "record_id,raster_id,number_gcps,gnd_truth_rmse\n"

    # Iterate over sub-folders...
    num_records = 0
    for folder in os.listdir(INPUT_MAPS_DIR):

        if not os.path.isdir(os.path.join(INPUT_MAPS_DIR, folder)):
            continue

        record_id = folder
        print(f"{num_records}")
        print(f"Processing record ID: {record_id}")

        # get final input path of geoTIFF
        # NOTE: assumed dir structure is INPUT_MAPS_DIR/<record_ID>/<record_ID>/<map_filename>.tif

        record_path = os.path.join(INPUT_MAPS_DIR, record_id, record_id)

        tif_paths = [f for f in Path(record_path).glob("*.tif")]
        if not tif_paths:
            print(f"WARNING! No TIFFs found for record ID {record_id}")
            continue
        if len(tif_paths) > 1:
            print(f"WARNING! Multiple TIFFs found for record ID {record_id}")
        tif_path = tif_paths[0]

        print(f"Loading TIFF {tif_path.name}...")
        with rasterio.open(tif_path) as tiff_data:

            gcps, crs_source = tiff_data.get_gcps()
            print(f"Number of GCPs found: {len(gcps)}")
            if len(gcps) == 0:
                continue
            crs_source = crs_source.to_string()

            crs_transform = None
            if crs_source != CRS_TARGET:
                print(f"Transforming CRS from {crs_source} to {CRS_TARGET}")
                crs_transform = Transformer.from_crs(
                    crs_source, CRS_TARGET, always_xy=True
                )

            csv_output = "raster_ID,row,col,NAD83_x,NAD83_y\n"
            # use the CDR cog_ID as the raster_id, if available
            raster_id = record2cog.get(record_id, record_id)
            print(f"Raster ID: {raster_id}")

            # get actual coords and x,y pixels of the GCPs to use as ground truth
            for gcp in gcps:
                coord_xy = (gcp.x, gcp.y)

                if crs_transform:
                    coord_xy_new = crs_transform.transform(coord_xy[0], coord_xy[1])
                    coord_xy = coord_xy_new
                    gcp.x = coord_xy[0]
                    gcp.y = coord_xy[1]

                # save GCP as a ground truth query point in CSV format
                csv_row = (
                    f"{raster_id},{gcp.row},{gcp.col},{coord_xy[0]},{coord_xy[1]}\n"
                )
                csv_output += csv_row

            # save CSV file
            csv_out_path = os.path.join(RESULTS_DIR_GEOREF, f"{raster_id}.csv")
            with open(csv_out_path, "w") as f_out:
                f_out.write(csv_output)

            # open image and re-save as a pixel-space TIFF
            img = Image.open(tif_path)
            tiff_out_path = os.path.join(RESULTS_DIR_GEOREF, f"{raster_id}.tif")
            img.save(tiff_out_path, compression="tiff_lzw")

            # sanity check: use the rasterio transform to score the ground truth query points
            rmse = score_query_points(gcps)

            print(f"RMSE of ground-truth transform: {rmse:.3f} km")
            summary_results_csv += f"{record_id},{raster_id},{len(gcps)},{rmse}\n"
            num_records += 1

        # save summary CSV file
        summary_csv_path = os.path.join(
            RESULTS_DIR_GEOREF, f"summary_georef_groundtruth.csv"
        )
        with open(summary_csv_path, "w") as f_out:
            f_out.write(summary_results_csv)

    print(f"Number of records: {num_records}")
    print("Done!")


if __name__ == "__main__":

    run()
