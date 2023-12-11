import os
import sys

from pipelines.geo_referencing.run_pipeline import process_folder
from pipelines.segmentation.run_pipeline import main


def run_pipeline(argv: list[str]) -> None:
    if argv[1] == "georef":
        folder = argv[2]
        process_folder(folder)
    else:
        main()


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    run_pipeline(sys.argv)
