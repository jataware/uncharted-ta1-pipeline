from pathlib import Path
import cdrc
import os
import argparse

token = os.getenv("CDR_API_TOKEN")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--cog_id", type=str, help="the COG ID", required=True)
    parser.add_argument("--output", type=Path, help="the output file path", default=".")
    parser.add_argument(
        "--system", type=str, help="the system name", default="uncharted"
    )
    parser.add_argument(
        "--version", type=str, help="the system version", default="0.0.4"
    )
    args = parser.parse_args()

    if token is None:
        raise ValueError("CDR_API_TOKEN is not set")

    # assert the supplied directory exists and create it if it doesn't
    if not args.output.is_dir():
        raise ValueError("Output path must be a directory")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    client = cdrc.CDRClient(token=token, output_dir=args.output)

    # Define the name of the layer to be created
    layer_name = f"{args.cog_id}_points"

    cog_id = args.cog_id
    system = args.system
    system_version = args.version

    client.build_cog_geopackages(
        cog_id=args.cog_id,
        feature_types=["point"],
        system_versions=[(args.system, args.version)],
        validated=None,
    )
