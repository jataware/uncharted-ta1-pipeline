import os

from util.coco import read_coco_file
from util.json import read_json_file

coco_cache = {}
model_cache = {}


def get_coco_annotations(path: str):
    # make sure the coco file has been loaded
    if path not in coco_cache:
        coco_json = read_coco_file(path)
        coco_cache[path] = coco_json
    return coco_cache[path]


def get_roi_coco(path: str, raster_id: str):
    # determine the image id for the raster
    coco_data = get_coco_annotations(path)
    image_id = get_image_id(coco_data, raster_id)
    if image_id < 0:
        return None

    # look up the relevant annotation (category 2, matching the image id)
    for a in coco_data["annotations"]:
        if a["category_id"] == 2 and a["image_id"] == image_id:
            polygon_flat = a["segmentation"][0]
            # build the x,y polygon
            return [
                (polygon_flat[i], polygon_flat[i + 1])
                for i in range(0, len(polygon_flat), 2)
            ]

    return None


def get_roi_model(path: str, raster_id: str):
    # assume there is a file with the raster id.json
    # TODO: HANDLE WHEN NO MODEL OUTPUT FOUND
    model_output = read_json_file(os.path.join(path, f"{raster_id}.json"))

    # find the map region polygon
    polygon = []
    for r in model_output:
        if r["class_label"] == "map":
            polygon = r["poly_bounds"]

    # map the inner lists as tuples for consistency
    return [(x[0], x[1]) for x in polygon]


def determine_roi(path: str, raster_id: str):
    # assume a file will be coco output and a directory will be model output
    if os.path.isfile(path):
        return get_roi_coco(path, raster_id)
    elif os.path.isdir(path):
        return get_roi_model(path, raster_id)

    return None


def get_image_id(coco_data, raster_id: str):
    for i in coco_data["images"]:
        if raster_id in i["file_name"]:
            return i["id"]
    return -1
