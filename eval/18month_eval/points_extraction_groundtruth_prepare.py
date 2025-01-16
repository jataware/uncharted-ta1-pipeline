from PIL import Image
import os, json
import rasterio
from rasterio.transform import from_gcps, AffineTransformer
from pathlib import Path
from pyproj import Transformer
from eval_utils import point_property_for_grouping, urlify
import glob
import fiona
from collections import defaultdict


# ---- ENV variables to edit
INPUT_MAPS_DIR = "<local/path/to/georeferencing/eval/geotiffs>"

INPUT_FEATURES_DIR = "local/path/to/eval/ground truth/feature extraction/dir"

COG_ID_TO_RECORD_JSON = "<local/path/to/ground_truth/cog_id_info.json>"

# -------

RESULTS_DIR_FE = "results/feature_extraction"
Image.MAX_IMAGE_PIXELS = 500000000


def run():

    print(f"*** Running points_extraction_groundtruth_prepare...")

    os.makedirs(RESULTS_DIR_FE, exist_ok=True)

    record2cog = {}
    if COG_ID_TO_RECORD_JSON:
        with open(COG_ID_TO_RECORD_JSON, "r") as fp:
            record2cog = json.load(fp)

    summary_results_csv = "record_id,raster_id,feature_id,num_features\n"

    # Iterate over sub-folders...
    num_records = 0
    bGo = False
    for folder in os.listdir(INPUT_FEATURES_DIR):

        curr_fe_path = os.path.join(INPUT_FEATURES_DIR, folder)
        if not os.path.isdir(curr_fe_path):
            continue

        record_id = folder
        print("")
        print(f"{num_records}")
        print(f"Processing record ID: {record_id}")

        # use the CDR cog_ID as the raster_id, if available
        raster_id = record2cog.get(record_id, record_id)
        print(f"Raster ID: {raster_id}")

        # get final input path of geoTIFF
        # assumed dir structure is INPUT_MAPS_DIR/<record_ID>/<record_ID>/<map_filename>.tif

        record_path = os.path.join(INPUT_MAPS_DIR, record_id, record_id)

        tif_paths = [f for f in Path(record_path).glob("*.tif")]
        if not tif_paths:
            print(f"WARNING! No TIFFs found for record ID {record_id}")
            continue
        if len(tif_paths) > 1:
            print(f"WARNING! Multiple TIFFs found for record ID {record_id}")
        tif_path = tif_paths[0]

        print(f"Loading TIFF {tif_path.name}...")
        crs_map = ""
        tiff_transform = None
        with rasterio.open(tif_path) as tiff_data:

            gcps, crs_map = tiff_data.get_gcps()
            print(f"Number of GCPs found: {len(gcps)}")
            if len(gcps) == 0:
                print("WARNING! No GCPs found. Skipping this map.")
                continue
            crs_map = crs_map.to_string()

            affine_transform = from_gcps(gcps)
            tiff_transform = AffineTransformer(affine_transform)

        # open image and re-save as a pixel-space TIFF
        img = Image.open(tif_path)
        img_wh = img.size
        tiff_out_path = os.path.join(RESULTS_DIR_FE, f"{raster_id}.tif")
        img.save(tiff_out_path, compression="tiff_lzw")

        # --- find and load any corresponding shapefiles for this TIFF
        shpfile_paths = glob.glob(
            os.path.join(curr_fe_path, "**", "*.shp"), recursive=True
        )

        print(f"Number of shapefiles found: {len(shpfile_paths)}")
        for shpfile_path in shpfile_paths:

            shpfilename = os.path.splitext(os.path.basename(shpfile_path))[0]
            print("")
            print(f"Shapefile: {shpfilename}")

            try:

                # Open the shapefile
                crs_transform = None
                with fiona.open(shpfile_path) as sh_file:
                    shf_crs = ""
                    if sh_file.crs:
                        shf_crs = sh_file.crs.to_string()
                    if shf_crs != crs_map:
                        print("WARNING!! Shape file and Map CRS codes do NOT match!!")
                        # continue  # skip for now
                        # create transform to go from shapefile to map CRS
                        crs_transform = Transformer.from_crs(
                            shf_crs, crs_map, always_xy=True
                        )

                    shf_properties = sh_file.schema["properties"].keys()  # type: ignore
                    shf_geometry = sh_file.schema["geometry"]  # type: ignore

                    if shf_geometry != "Point":
                        print(
                            f"Skipping shapefile {shpfilename}. Geometry = {shf_geometry}"
                        )
                        continue

                    groupby_key = point_property_for_grouping(shf_properties)

                    # Iterate over the feature records
                    feats = defaultdict(list)
                    for rec in sh_file:

                        feat_key = shpfilename
                        if groupby_key:
                            groupby_prop = rec["properties"].get(groupby_key, "")
                            if groupby_prop:
                                feat_key += "_" + groupby_prop
                            # sometimes "vertical" or "horizontal" bedding or foliation points
                            # are grouped with inclined points, but with a dip angle of 90
                            groupby_suffix = ""
                            dip_angle = 0
                            if "Inclinatio" in rec["properties"]:
                                dip_angle = rec["properties"]["Inclinatio"]
                            elif "Dip" in rec["properties"]:
                                dip_angle = rec["properties"]["Dip"]
                            if dip_angle == 90:
                                groupby_suffix = "_vertical"
                            elif dip_angle == 0:
                                groupby_suffix = "_horizontal"
                            feat_key += groupby_suffix

                        # get feature coords (world CRS-based coords), convert to pixels, and save to dict
                        feat_coords = rec["geometry"].coordinates
                        if crs_transform:
                            feat_coords_new = crs_transform.transform(
                                feat_coords[0], feat_coords[1]
                            )
                            feat_coords = feat_coords_new

                        # convert to pixel coords
                        pxl_y, pxl_x = tiff_transform.rowcol(
                            feat_coords[0], feat_coords[1]
                        )
                        feats[feat_key].append((pxl_x, pxl_y))

                # save results
                print(f"Number of feature types: {len(feats.keys())}")
                for feat_key, feat_list in feats.items():
                    print(
                        f"Feature type: {feat_key}, number of points: {len(feat_list)}"
                    )

                    summary_results_csv += (
                        f"{record_id},{raster_id},{feat_key},{len(feat_list)}\n"
                    )

                    # save JSON results and image bitmask
                    feat_filen = raster_id + "_" + feat_key
                    feat_filen = urlify(feat_filen)

                    with open(
                        os.path.join(RESULTS_DIR_FE, feat_filen + ".json"), "w"
                    ) as f_out:
                        json.dump(feat_list, f_out)

            except Exception as ex:
                print(f"Exception processing shapefile!! -- {repr(ex)}")

        num_records += 1

    # save summary CSV file
    summary_csv_path = os.path.join(RESULTS_DIR_FE, f"summary_fe_groundtruth.csv")
    with open(summary_csv_path, "w") as f_out:
        f_out.write(summary_results_csv)

    print(f"Number of records: {num_records}")
    print("Done!")


if __name__ == "__main__":

    run()
