import os
import hashlib
from typing import Dict

LOCAL_CACHE_PATH = "~/.points"


def filename_to_id(filename: str) -> int:
    """
    Hash filename and modification time to create a unique id.
    """
    mod_time = os.path.getmtime(filename)
    unique_string = filename + str(mod_time)
    hash_object = hashlib.md5(unique_string.encode())
    return int(hash_object.hexdigest()[:8], 16)


def filter_coco_images(coco_dict: Dict) -> Dict:
    # Filter out images that have no annotations.
    filtered_coco_dict = {
        "images": [],
        "annotations": [],
        "categories": coco_dict["categories"],
    }
    image_ids = set(anno["image_id"] for anno in coco_dict["annotations"])
    for image in coco_dict["images"]:
        if image["id"] in image_ids:
            filtered_coco_dict["images"].append(image)
    for anno in coco_dict["annotations"]:
        if anno["image_id"] in image_ids:
            filtered_coco_dict["annotations"].append(anno)
    print(
        f'Filtering {len(coco_dict["images"]) - len(filtered_coco_dict["images"])} out of {len(coco_dict["images"])} images without annotations'
    )
    return filtered_coco_dict
