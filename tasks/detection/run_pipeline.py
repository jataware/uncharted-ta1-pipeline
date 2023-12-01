import argparse
import json
from PIL import Image

Image.MAX_IMAGE_PIXELS = 933120000

from detection.entities import Pipeline, MapImage
from detection.tiling import Tiler, Untiler
from detection.point_detector import PointDirectionPredictor, YOLOPointDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input image."
    )
    parser.add_argument(
        "--model_ckpt", type=str, required=True, help="Path to model checkpoint."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output json."
    )
    args = parser.parse_args()

    pipeline = Pipeline(
        [
            Tiler(),
            YOLOPointDetector(ckpt=args.model_ckpt),
            Untiler(),
            PointDirectionPredictor(),
        ]
    )
    map_image = MapImage(path=args.input_path, image=Image.open(args.input_path))
    labeled_map = pipeline.process(map_image)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_map.serialize(), f)


if __name__ == "__main__":
    main()
