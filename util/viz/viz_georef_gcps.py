from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict
import os
from pathlib import Path

# -------------
# Plot pxl locations and geo-coord labels for georeferencing GCPs


RESULTS_DIR = "results_viz/"
BBOX_LEN_DEFAULT = 150
CDR_JSON_SUFFIX = ""

Image.MAX_IMAGE_PIXELS = 500000000

FONT_SIZE = 24
COLOUR = "red"
FILL_COLOUR = (128, 0, 128, 128)
TEXT_COLOUR = "darkmagenta"
LINE_WIDTH = 6


def run(input_dir: str, json_gcps_dir: str):
    print(f"*** Running Viz Georef GCPs on image path : {input_dir}")

    input_path = Path(input_dir)
    json_gcps_path = Path(json_gcps_dir)

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

        json_path = json_gcps_path / f"{img_filen}.json"
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

        gcps = data[0]["map"]["projection_info"]["gcps"]
        if not gcps:
            print("Oooops!! No GCPs! Skipping")
            continue
        print(f"Visualizing {len(gcps)} GCPs")

        # Create a drawing object to overlay bounding boxes
        draw = ImageDraw.Draw(img, mode="RGBA")

        # Draw bounding boxes from the labels onto the image
        # create a color map for each unique class_name
        counts_per_class = defaultdict(int)
        font = ImageFont.load_default(FONT_SIZE)

        for gcp in gcps:
            lonlat = gcp["map_geom"]
            xy = gcp["px_geom"]
            x1 = int(xy[0] - BBOX_LEN_DEFAULT / 2)
            y1 = int(xy[1] - BBOX_LEN_DEFAULT / 2)
            box_coords = (
                x1,
                y1,
                x1 + BBOX_LEN_DEFAULT,
                y1 + BBOX_LEN_DEFAULT,
            )

            draw.rectangle(
                box_coords, outline=COLOUR, fill=FILL_COLOUR, width=LINE_WIDTH
            )

            coord_text = f"lon:{lonlat[0]:.3f}, lat:{lonlat[1]:.3f}"
            draw.text(
                (x1, y1 - FONT_SIZE * 1.5),
                coord_text,
                fill=TEXT_COLOUR,
                font=font,
                font_size=FONT_SIZE,
            )

        print(f"Number of GCPs: {len(gcps)}")

        img.save(
            os.path.join(RESULTS_DIR, img_filen + "_gcps_viz.jpg")
        )  # save output to a new jpg file

    print("Done!")


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--input_path", type=str, help="path to input tiffs")
    args.add_argument(
        "--json_gcps_dir",
        type=str,
        help="path to uncharted georef GCP json results",
        default="",
    )
    p = args.parse_args()

    run(p.input_path, p.json_gcps_dir)
