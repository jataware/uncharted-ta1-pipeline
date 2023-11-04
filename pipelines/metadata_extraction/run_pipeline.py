import argparse
from pathlib import Path
from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractorPipeline,
)


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    p = parser.parse_args()

    # run the extraction pipeline
    pipeline = MetadataExtractorPipeline(p.input, p.output, p.workdir, p.verbose)
    pipeline.run()


if __name__ == "__main__":
    main()
