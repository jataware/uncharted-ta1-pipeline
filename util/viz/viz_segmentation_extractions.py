from argparse import ArgumentParser
from PIL import Image, ImageDraw
import json
import os
from pathlib import Path


# -------------
# Plot Segmentation extractions on an image


RESULTS_DIR = "results_viz/"

Image.MAX_IMAGE_PIXELS = 500000000

FONT_SIZE = 16

category2colour = {
    "cross_section": "limegreen",
    "legend_points_lines": "blue",
    "legend_polygons": "red",
    "map": "yellow",
}


def run(input_path: Path, json_pred_path: Path):
    print(f"*** Running viz segmentations on image path : {input_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    input_files = []
    if input_path.is_dir():
        # collect the ids of the files in the directory
        input_files = [file for file in input_path.glob("*.tif")]
    else:
        input_files = [input_path]

    for img_path in input_files:

        img_filen = Path(img_path).stem
        json_path = json_pred_path / f"{img_filen}_map_segmentation.json"
        print(f"---- {img_filen}")

        data = json.load(open(json_path))
        img = Image.open(img_path)

        # Create a drawing object to overlay bounding boxes
        draw = ImageDraw.Draw(img, mode="RGBA")

        segments = data.get("segments", [])
        if not segments:
            print("Oooops!! No predictions! Exiting")
            return
        print(f"Visualizing {len(segments)} segments")

        for segment in segments:
            confidence = segment["confidence"]
            label = segment["class_label"]
            poly_bounds = segment["poly_bounds"]
            xy = []
            for p in poly_bounds:
                p1 = (int(p[0]), int(p[1]))
                xy.append(p1)

            c = category2colour[label]

            draw.polygon(xy, outline=c, width=30)

            print(f"   {label}: {confidence}")

        img.save(
            os.path.join(RESULTS_DIR, img_filen + "_segmentation_viz.jpg")
        )  # save output to a new jpg file

    print("Done!")


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--input_path", type=Path, help="path to input tiffs")
    args.add_argument(
        "--json_pred_path",
        type=Path,
        help="path to uncharted segementation extraction json results",
    )
    p = args.parse_args()

    run(p.input_path, p.json_pred_path)
