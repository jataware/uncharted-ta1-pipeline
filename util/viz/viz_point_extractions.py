from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict

import math
import os
from pathlib import Path

from schema.cdr_schemas.cdr_responses.features import PointExtractionResponse
from tasks.point_extraction.label_map import YOLO_TO_CDR_LABEL
from tasks.point_extraction.point_extractor_utils import rotated_about

# -------------
# Plot Point Extraction location and orientation bboxes on an image


RESULTS_DIR = "results_viz/"
BBOX_LEN_DEFAULT = 90
CDR_JSON_SUFFIX = ""

Image.MAX_IMAGE_PIXELS = 500000000

FONT_SIZE = 16
DRAW_ROTATED_RECTS = True

# from https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
# def HSVToRGB(h, s, v):
#     (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
#     return (int(255 * r), int(255 * g), int(255 * b))

# def getDistinctColors(n):
#     huePartition = 1.0 / (n + 1)
#     return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n))

class2colour = {
    "strike_and_dip": "red",
    "horizontal_bedding": "blue",
    "overturned_bedding": "green",
    "vertical_bedding": "orange",
    "inclined_foliation": "darkmagenta",
    "inclined_foliation_igneous": "limegreen",
    "vertical_foliation": "springgreen",
    "vertical_joint": "turquoise",
    "gravel_borrow_pit": "hotpink",
    "mine_shaft": "darkviolet",
    "prospect": "deepskyblue",
    "mine_tunnel": "tomato",
    "mine_quarry": "limegreen",
    "sink_hole": "goldenrod",
    "lineation": "violet",
    "drill_hole": "cyan",
}

cdr_to_yolo = {v: k for k, v in YOLO_TO_CDR_LABEL.items()}


def parse_cdr_features(data: list):

    pt_labels = []
    for d in data:
        pt_label = {}
        rec = PointExtractionResponse(**d)
        bbox = rec.px_bbox
        if bbox[0] == bbox[2]:
            # 0th size bbox (x axis)
            pt_label["x1"] = int(bbox[0] - BBOX_LEN_DEFAULT / 2)
            pt_label["x2"] = int(bbox[2] + BBOX_LEN_DEFAULT / 2)
        else:
            pt_label["x1"] = int(rec.px_bbox[0])
            pt_label["x2"] = int(rec.px_bbox[2])

        if bbox[1] == bbox[3]:
            # 0th size bbox (y axis)
            pt_label["y1"] = int(bbox[1] - BBOX_LEN_DEFAULT / 2)
            pt_label["y2"] = int(bbox[3] + BBOX_LEN_DEFAULT / 2)
        else:
            pt_label["y1"] = int(rec.px_bbox[1])
            pt_label["y2"] = int(rec.px_bbox[3])
        pt_label["score"] = rec.confidence if rec.confidence else 0.5
        pt_label["direction"] = rec.dip_direction
        pt_label["dip"] = rec.dip

        label = (rec.legend_item.get("label", "") if rec.legend_item else "").strip()
        if label in cdr_to_yolo:
            label = cdr_to_yolo[label]
        pt_label["class_name"] = label

        pt_labels.append(pt_label)
    return pt_labels


def run(input_dir: str, json_pred_dir: str, cdr_json_pred_dir: str):
    print(f"*** Running viz point extractions on image path : {input_dir}")

    input_path = Path(input_dir)
    cdr_json_pred_path = Path(cdr_json_pred_dir)
    json_pred_path = Path(json_pred_dir)
    parse_cdr_preds = True if cdr_json_pred_dir else False

    os.makedirs(RESULTS_DIR, exist_ok=True)

    input_files = []
    if input_path.is_dir():
        # collect the ids of the files in the directory
        input_files = [file for file in input_path.glob("*.tif")]
    else:
        input_files = [input_path]

    for img_path in input_files:

        img_filen = Path(img_path).stem
        # --- TEMP code needed to run with contest dir-based data
        if (
            img_filen.endswith("_pt")
            or img_filen.endswith("_poly")
            or img_filen.endswith("_line")
            or img_filen.endswith("_point")
        ):
            print(f"Skipping {img_filen}")
            continue

        if parse_cdr_preds:
            json_path = cdr_json_pred_path / f"{img_filen}{CDR_JSON_SUFFIX}.json"
        else:
            json_path = json_pred_path / f"{img_filen}_point_extraction.json"
        print(f"---- {img_filen}")

        try:
            data = json.load(open(json_path))
            img = Image.open(img_path)

            # Create a drawing object to overlay bounding boxes
            draw = ImageDraw.Draw(img, mode="RGB")
        except Exception as e:
            # try converting mode to RGBA
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img, mode="RGB")

        if parse_cdr_preds:
            print("Parsing CDR records")
            labels = parse_cdr_features(data)

        else:
            labels = data.get("labels", [])
            if not labels:
                print("Oooops!! No predictions! Skipping")
                continue
        print(f"Visualizing {len(labels)} point predictions")

        # Create a drawing object to overlay bounding boxes
        draw = ImageDraw.Draw(img, mode="RGBA")

        # Draw bounding boxes from the labels onto the image
        # create a color map for each unique class_name
        counts_per_class = defaultdict(int)
        font = ImageFont.load_default(FONT_SIZE)

        for label in labels:
            box_coords = (label["x1"], label["y1"], label["x2"], label["y2"])
            class_name = label["class_name"]
            score = label["score"]
            if class_name not in class2colour:
                print(
                    f"Oooopps!! class name not recognized: {class_name}. Plotting with yellow"
                )
                colour = "yellow"
            else:
                colour = class2colour[class_name]
            if DRAW_ROTATED_RECTS and label["direction"]:
                rot_angle_compass = float(label["direction"])
                # convert angle compass angle convention (CW with 0 deg at top)
                # regular 'trig' angle convention (CCW with 0 to the right)
                rot_angle = 270 - rot_angle_compass
                if rot_angle < 0:
                    rot_angle += 360
                xc = (label["x1"] + label["x2"]) / 2.0
                yc = (label["y1"] + label["y2"]) / 2.0
                square_vertices = [
                    (label["x1"], label["y1"]),
                    (label["x2"], label["y1"]),
                    (label["x2"], label["y2"]),
                    (label["x1"], label["y2"]),
                ]
                square_vertices = [
                    rotated_about(x, y, xc, yc, math.radians(-rot_angle))
                    for x, y in square_vertices
                ]
                draw.polygon(square_vertices, outline=colour, width=3)
            else:
                draw.rectangle(box_coords, outline=colour, width=3)

            class_text = class_name.replace("_", " ")
            draw.text(
                (label["x1"], label["y1"] - FONT_SIZE),
                class_text,
                fill=colour,
                font=font,
                font_size=FONT_SIZE,
            )

            angles_text = ""
            # don't write strike angles for now (shown by rotated rect)
            # if label["direction"]:
            #    angles_text = f'angle: {int(label["direction"])}'
            if label["dip"]:
                if angles_text:
                    angles_text += ", "
                angles_text += f'dip: {int(label["dip"])}'
            if angles_text:
                draw.text(
                    (label["x1"], label["y2"]),
                    angles_text,
                    fill=colour,
                    font=font,
                    font_size=FONT_SIZE,
                )

            counts_per_class[class_name] += 1

        print("--------")
        print("Detections per class:")
        for label, num in counts_per_class.items():
            print(f"{label}:  {num}")

        img.save(
            os.path.join(RESULTS_DIR, img_filen + "_points_viz.jpg")
        )  # save output to a new jpg file

    print("Done!")


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--input_path", type=str, help="path to input tiffs")
    args.add_argument(
        "--json_pred_path",
        type=str,
        help="path to uncharted point extraction json results",
        default="",
    )
    args.add_argument(
        "--cdr_json_pred_path",
        type=str,
        help="path to uncharted point extraction json results in CDR format",
        default="",
    )
    p = args.parse_args()

    run(p.input_path, p.json_pred_path, p.cdr_json_pred_path)
