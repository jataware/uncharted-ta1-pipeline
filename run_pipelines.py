import os
import sys

from pipelines.geo_referencing.run_pipeline import process_folder


def run_pipeline(argv: list[str]) -> None:
    if argv[1] == "georef":
        folder = argv[2]
        process_folder(folder)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    run_pipeline(sys.argv)
