import argparse
from pathlib import Path
from pipelines.metadata_extraction.scorer import Scorer


def main():
    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default="")
    parser.add_argument("--verbose", default=False)
    parser.add_argument("--approx_long", default=False)
    p = parser.parse_args()

    scorer = Scorer(p.truth, p.predictions, p.approx_long, p.verbose)
    scorer.score()
    scorer.print_scores()


if __name__ == "__main__":
    main()
