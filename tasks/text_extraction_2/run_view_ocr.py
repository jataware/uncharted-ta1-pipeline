import argparse
import os
from pathlib import Path
import metadata_extraction.task.ocr_util as ocr_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_input", type=Path, required=True)
    parser.add_argument("--ocr_input", type=Path, required=True)
    p = parser.parse_args()

    pil_image = ocr_util.load_pil_image(p.image_input)
    pil_image = ocr_util.condition_pil_image(pil_image)
    texts = ocr_util.load_ocr_output(p.ocr_input)

    ocr_util.display_ocr_results(texts, pil_image, color="magenta")


if __name__ == "__main__":
    main()
