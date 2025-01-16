from PIL import Image
import os, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from fe_metrics_utils import calc_f1_score_pts

"""
Script to prepare points extraction ground truth results for the 18-month self-evaluation
Input data is assumed to follow the directory structure in the 18-month eval data.tar.gz package from Mitre
See README.md for more info
"""

# ---- ENV variables to edit
INPUT_LARA_POINTS_DIR = "<path/to/lara/points_extraction/json/results/dir"

INPUT_GNDTRUTH_FEATURES_DIR = "<path/to/ground_truth/point/features/json/dir>"

INPUT_MAPS_DIR = "<local/path/to/georeferencing/eval/geotiffs>"

# -------------

RESULTS_DIR_FE = "results/feature_extraction"
Image.MAX_IMAGE_PIXELS = 500000000

POINT_LABELS = [
    "inclined_bedding",
    "horizontal_bedding",
    "overturned_bedding",
    "vertical_bedding",
    "inclined_foliation_metamorphic",
    "inclined_foliation_igneous",
    "vertical_foliation",
    "vertical_joint",
    "sink_hole",
    "lineation",
    "drill_hole",
    "pit",
    "mine_shaft",
    "prospect",
    "mine_tunnel",
    "quarry",
]


def run():

    print(f"*** Running fe_calc_metrics...")

    os.makedirs(RESULTS_DIR_FE, exist_ok=True)

    # Iterate over LARA cdr-format json results files...
    lara_json_paths = [f for f in Path(INPUT_LARA_POINTS_DIR).glob("*_cdr.json")]

    num_records = 0
    metrics_per_file = {}
    for lara_file in lara_json_paths:

        raster_id = os.path.splitext(os.path.basename(lara_file))[0]
        raster_id = raster_id.split("_")[0]
        print("")
        print(f"Processing cog ID: {raster_id}")

        img = Image.open(os.path.join(INPUT_MAPS_DIR, f"{raster_id}.tif"))
        img_w_h = img.size

        lara_pts = defaultdict(list)
        with open(lara_file, "r") as f_lara_in:
            lara_data = json.load(f_lara_in)
            pt_features = lara_data.get("point_feature_results", [])
            for feats in pt_features:
                pt_label = feats["name"]
                pts = []
                for feat in feats["point_features"]["features"]:
                    pt_xy = feat["geometry"]["coordinates"]
                    bbox = feat["properties"]["bbox"]
                    pts.append(pt_xy)

                print(f"point type: {pt_label}, {len(pts)}")
                lara_pts[pt_label].extend(pts)

        # find/load all gnd truth json files for this COG
        gt_json_paths = [
            f for f in Path(INPUT_GNDTRUTH_FEATURES_DIR).glob(f"{raster_id}_*.json")
        ]

        gt_pts = defaultdict(list)
        print(f"Loading {len(gt_json_paths)} ground truth json records for this COG...")
        for gt_path in gt_json_paths:

            gt_file = os.path.splitext(os.path.basename(gt_path))[0]
            print(f"GT file: {gt_file}")

            # get feature label
            # assume <cog>_<other descriptors>__<point ontology type>.json

            splits = gt_file.split("__")
            if len(splits) < 1:
                print("Point label not found. Skipping")
                continue
            pt_label = splits[-1]
            if pt_label not in POINT_LABELS:
                print(f"Point label type {pt_label} NOT found in ontology. Skipping.")
                continue
            print(f"Point label type: {pt_label}")

            with open(gt_path, "r") as f_gt_in:
                gt_data = json.load(f_gt_in)
                gt_pts[pt_label].extend(gt_data)

        if not gt_pts:
            print("No ground truth ontology-based points for this COG. Skipping.")
            continue

        print("GT ontology-based points:")
        for pt_label, pts in gt_pts.items():
            print(f"point type: {pt_label}, {len(pts)}")

        # ---- compute metrics per feature per map
        results = calc_f1_score_pts(gt_pts, lara_pts, img_w_h)
        # Results are a list of (feature_label,num_gt_pts,num_pred_pts,precision,recall,f1_score)
        metrics_per_file[raster_id] = results

        num_records += 1

    # save summary CSV file
    summary_results_csv = (
        "raster_id,feature_label,num_gt_pts,num_pred_pts,precision,recall,f1_score\n"
    )
    f1_scores = []
    for raster_id, results in metrics_per_file.items():
        for pt_label, num_gt_pts, num_pred_pts, precision, recall, f1_score in results:
            this_row = f"{raster_id},{pt_label},{num_gt_pts},{num_pred_pts},{precision},{recall},{f1_score}\n"
            summary_results_csv += this_row
            f1_scores.append(f1_score)

    # # save summary CSV file
    summary_csv_path = os.path.join(RESULTS_DIR_FE, f"summary_fe_results.csv")
    with open(summary_csv_path, "w") as f_out:
        f_out.write(summary_results_csv)

    if f1_scores:
        a = np.array(f1_scores)
        print(f"POINTS, F1 Decile scores (for {len(f1_scores)} features)")
        print("{:.3f}".format(np.percentile(a, 10)))
        print("{:.3f}".format(np.percentile(a, 20)))
        print("{:.3f}".format(np.percentile(a, 30)))
        print("{:.3f}".format(np.percentile(a, 40)))
        print("{:.3f}".format(np.percentile(a, 50)))
        print("{:.3f}".format(np.percentile(a, 60)))
        print("{:.3f}".format(np.percentile(a, 70)))
        print("{:.3f}".format(np.percentile(a, 80)))
        print("{:.3f}".format(np.percentile(a, 90)))

    print(f"Number of records: {num_records}")
    print("Done!")


if __name__ == "__main__":

    run()
