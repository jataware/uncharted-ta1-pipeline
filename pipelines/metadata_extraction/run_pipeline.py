import argparse
from pathlib import Path
from metadata_extraction_pipeline import MetadataExtractorPipeline


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    p = parser.parse_args()

    # run the extraction pipeline
    pipeline = MetadataExtractorPipeline(p.input, p.output, p.workdir, p.verbose)
    # pipeline = MetadataExtractorPipeline(
    #     Path("/fdata/data/critical_maas/pipeline_test_subset_2"),
    #     Path("pipelines/metadata_exraction/working/output"),
    #     Path("pipelines/metadata_exraction/working"),
    #     True,
    # )
    pipeline.run()


if __name__ == "__main__":
    main()
