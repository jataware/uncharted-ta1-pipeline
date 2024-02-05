from argparse import ArgumentParser
import subprocess
import csv
from typing import List, Tuple
from pathlib import Path
from numpy import std
from osgeo import gdal, osr


def write_csv_headers(georef_input_file: Path, georef_truth_file: Path):
    # write the headers to the csv files
    with open(georef_input_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["raster_ID", "row", "col"])

    with open(georef_truth_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["raster_ID", "row", "col", "NAD83_x", "NAD83_y"])


def write_csv_files(
    georef_input_file: Path,
    georef_truth_file: Path,
    map_id: str,
    pixel_coords: List[Tuple[int, int]],
    geo_coords: List[Tuple[float, float]],
):
    # write the transformed gcps to a csv file
    with open(georef_input_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for pixel, geo in zip(pixel_coords, geo_coords):
            writer.writerow([map_id, pixel[1], pixel[0]])

    with open(georef_truth_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for pixel, geo in zip(pixel_coords, geo_coords):
            writer.writerow([map_id, pixel[1], pixel[0], geo[0], geo[1]])


def extract_pixel_gcps(file_path: Path) -> List[Tuple[int, int]]:
    # Open the GeoTIFF file and extract the GCPs
    ds = gdal.Open(str(file_path))
    gcps = ds.GetGCPs()
    return [(round(gcp.GCPPixel), round(gcp.GCPLine)) for gcp in gcps]


def transform_pixel_gcps(
    file_path: Path, gcp_coordinates: List[Tuple[int, int]], crs_epsg=4269
) -> List[Tuple[float, float]]:

    # Convert the pixel/line coordinates to a string
    coords_str = "\n".join([f"{gcp[0]} {gcp[1]}" for gcp in gcp_coordinates])

    # Run gdaltransform to convert the pixel/line coordinates to geographic coordinates
    result = subprocess.run(
        ["gdaltransform", str(file_path), "-t_srs", f"EPSG:{str(crs_epsg)}"],
        check=False,
        input=coords_str,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
    )

    # Parse the output of gdaltransform
    geo_coords: List[Tuple[float, float]] = []
    for line in result.stdout.splitlines():
        x_str, y_str, _ = line.split(" ")
        geo_coords.append((float(x_str), float(y_str)))

    return geo_coords


# main
def main():
    # use argparse to get the file path
    args = ArgumentParser()
    args.add_argument("--input_path", type=Path)
    args.add_argument("--output_name", type=str, default="georef")
    args.add_argument("--output_dir", type=Path, default=Path.cwd())
    args.add_argument("--crs_epsg", type=int, default=4269)

    p = args.parse_args()
    input_path: Path = p.input_path

    input_files: List[Path] = []
    if input_path.is_dir():
        # collect the ids of the files in the directory
        input_files = [file for file in input_path.glob("*.tif")]
    else:
        input_files = [input_path]

    georef_input_file = Path(p.output_dir / f"{p.output_name}_input.csv")
    georef_truth_file = Path(p.output_dir / f"{p.output_name}_truth.csv")

    # write the headers to the csv files
    write_csv_headers(georef_input_file, georef_truth_file)

    # iterate through the input files and append the pixel and geocoords to the output files
    for input_file in input_files:
        map_id = input_file.stem

        # transform the source gcps
        pixel_coords = extract_pixel_gcps(input_file)

        # transform the pixel gcps to geographic coordinates
        geo_coords = transform_pixel_gcps(input_file, pixel_coords, p.crs_epsg)

        # write the transformed gcps to a csv file
        write_csv_files(
            georef_input_file, georef_truth_file, map_id, pixel_coords, geo_coords
        )


if __name__ == "__main__":
    main()
