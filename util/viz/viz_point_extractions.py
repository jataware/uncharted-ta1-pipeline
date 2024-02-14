from PIL import Image, ImageDraw, ImageFont
import json
from collections import defaultdict

# import colorsys, random
import math
from matplotlib import colors as mpl_colors
import os
from pathlib import Path

# -------------
# Plot Point Extraction location and orientation bboxes on an image

IMG_PATH = "<path/to/images/tiffs>"

JSON_PRED_PATH = "<path/to/uncharted/point/extraction/json/results>"

RESULTS_DIR = "results_viz/"

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
    "vertical_foliation": "springgreen",
    "vertical_joint": "turquoise",
    "gravel_borrow_pit": "hotpink",
    "mine_shaft": "darkviolet",
    "prospect": "deepskyblue",
    "mine_tunnel": "tomato",
    "mine_quarry": "limegreen",
    "sink_hole": "goldenrod",
}


# finds the straight-line distance between two points
def distance(ax, ay, bx, by):
    return math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)


# from https://stackoverflow.com/questions/34747946/rotating-a-square-in-pil
# rotates point `A` about point `B` by `angle` radians clockwise.
def rotated_about(ax, ay, bx, by, angle):
    radius = distance(ax, ay, bx, by)
    angle += math.atan2(ay - by, ax - bx)
    return (round(bx + radius * math.cos(angle)), round(by + radius * math.sin(angle)))


def run():
    print(f"*** Running viz point extractions on image path : {IMG_PATH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    input_path = Path(IMG_PATH)
    input_files = []
    if input_path.is_dir():
        # collect the ids of the files in the directory
        input_files = [file for file in input_path.glob("*.tif")]
    else:
        input_files = [input_path]

    for img_path in input_files:

        img_filen = Path(img_path).stem
        json_path = Path(JSON_PRED_PATH) / f"{img_filen}_point_extraction.json"
        print(f"---- {img_filen}")

        data = json.load(open(json_path))
        img = Image.open(img_path)

        labels = data.get("labels", [])
        if not labels:
            print("Oooops!! No predictions! Exiting")
            return
        print(f"Visualizing {len(labels)} point predictions")

        # Create a drawing object to overlay bounding boxes
        draw = ImageDraw.Draw(img, mode="RGBA")

        # Draw bounding boxes from the labels onto the image
        # create a color map for each unique class_name
        counts_per_class = defaultdict(int)
        font = ImageFont.load_default(FONT_SIZE)

        for label in data["labels"]:
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
        )  # save output to a new png file

    print("Done!")


if __name__ == "__main__":
    run()
