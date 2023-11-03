import argparse
import requests
from pathlib import Path

# parse input file from command line
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
p = parser.parse_args()

# load the input file csv file
with open(p.input, "r") as input_file:
    for line in input_file:
        # skip header of the csv file
        if line.startswith("proddesc"):
            continue

        # split the line into a list
        line_list = line.split(",")
        # get the map id
        map_id = line_list[0]
        print(map_id)
        cloudfront_url = (
            f"https://d39ptu40l71xc2.cloudfront.net/pub/ngmdb/{map_id}_1.tif"
        )
        # fetch the map from cloudfront using http get
        response = requests.get(cloudfront_url)

        # check for 200 response
        if response.status_code == 200:
            # save the map to the local directory
            with open(f"{p.output}/{map_id}.tif", "wb") as map_file:
                map_file.write(response.content)
        else:
            print(f"error fetching map {map_id}")
