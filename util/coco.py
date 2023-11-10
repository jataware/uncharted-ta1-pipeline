
import json

def read_coco_file(path):
    # read it as json
    raw_file = open(path)
    return json.load(raw_file)