from detection.label_map import LABEL_MAPPING
from detection.utils import filename_to_id, filter_coco_images

import numpy as np
import os
from PIL import Image
from scipy.ndimage import label as scipy_label
from tqdm import tqdm
from typing import Dict, List, Union

Image.MAX_IMAGE_PIXELS = 500000000


class CompetitionCOCOLoader:

    """
    Assumptions to be made before using this loader:
    - Used to convert original competition training & validation datasets to COCO format.
    - Each map has a .json file with the same name.
    - Point features have bitmasks with a _pt indicator prior to the .tif extension.
    - Bitmasks are named with the same name as the map they are associated with, followed by the label name.
    - Bitmasks are stored in the same directory as the map they are associated with.
    """

    BAD_JSON = [
        "AZ_PrescottNF_basemap.json"
    ]  # Manually filter files with .json labels that dont map to any raster map.

    coco_dict = {"images": [], "annotations": [], "categories": []}

    def _fetch_map_paths(self, data_dir: str) -> List[str]:
        # This assumes that all files with the .json extension are input map files.
        paths = [
            os.path.join(data_dir, file.replace(".json", ".tif"))
            for file in os.listdir(data_dir)
            if ".json" in file and file not in self.BAD_JSON
        ]
        return paths

    @staticmethod
    def _fetch_bitmask_paths(data_dir: str, map_paths) -> Dict[str, List[str]]:
        out_dict = {}
        map_filenames = sorted(
            [os.path.basename(p).split(".")[0] for p in map_paths],
            key=len,
            reverse=True,
        )  # Sort by length so the most specific match is used first. We'll use this logic to remove duplicate bitmask matches.
        all_files = os.listdir(data_dir)
        bitmask_paths = [
            os.path.join(data_dir, f)
            for f in all_files
            if f.split(".")[0] not in map_filenames
        ]
        for map_name in map_filenames:
            # check if map_name appears as a substring in any other map files.
            match_candidates = [path for path in bitmask_paths if map_name in path]
            flat_values = [i for j in out_dict.values() for i in j]
            for match in match_candidates:
                if match in flat_values:
                    # This is really bad, but its temporary. We ought to just preprocess and store the training/validation data somewhere.
                    # The logic here is that since we are matching longer maps first, we can remove any matches already in the output dict since we know that the longest match is the most specific.
                    print(
                        f"Removing duplicate match: {match} from map: {map_name}, since it is already in the output dict."
                    )
                    match_candidates.remove(match)
            out_dict[map_name] = match_candidates
        return out_dict

    @staticmethod
    def _filter_bitmask_paths(
        bitmask_paths: List[str], label_names: List[str]
    ) -> List[str]:
        return [i for i in bitmask_paths if any(label in i for label in label_names)]

    @staticmethod
    def _fetch_map_metadata(map_path: str) -> Dict[str, Union[str, int]]:
        return {
            "id": filename_to_id(map_path),
            "file_name": map_path.split("/")[-1].split(".")[0],
            "coco_url": map_path,
        }

    @staticmethod
    def construct_bounding_boxes(
        bitmask: Union[str, np.array], bounding_box_size: List[int] = [50, 50]
    ) -> List[List[int]]:
        # this is slow, due to massive size of bitmask files.. the points labeled in the .json files are the legend icons.
        # not sure what to do about this.

        if isinstance(bitmask, str):
            bitmask = np.array(Image.open(bitmask))
        bitmask = bitmask > 0

        labeled, num_features = scipy_label(bitmask)
        bounding_boxes = []

        for i in range(
            1, num_features + 1
        ):  # Iterate through each feature in the bitmask. Each feature corresponds to a point equal to 1.
            component = labeled == i

            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # For height
            diff_r = bounding_box_size[0] - (rmax - rmin)
            if diff_r > 0:
                rmin = max(0, rmin - diff_r // 2)  # don't get negative values
                rmax = rmax + diff_r // 2

            # For width
            diff_c = bounding_box_size[1] - (cmax - cmin)
            if diff_c > 0:
                cmin = max(0, cmin - diff_c // 2)  # don't exceed the left edge
                cmax = cmax + diff_c // 2

            bounding_boxes.append(
                [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]
            )  # x, y, width, height

        return bounding_boxes

    def process(
        self,
        raster_dir: str,
        bbox_shape: List[int] = [50, 50],
    ):
        label_map = LABEL_MAPPING

        maps = self._fetch_map_paths(raster_dir)
        map_metadata = [self._fetch_map_metadata(_map) for _map in maps]
        label_names = list(label_map.keys())
        bitmask_reverse_index = {}
        for k, v in label_map.items():
            for i in v:
                bitmask_reverse_index[i] = k
        label_aliases = set(
            [i for j in label_map.values() for i in j]
        )  # account for variation in bitmask file names, flatten all values in mapping.
        categories = [
            {"id": i, "name": label_names[i]} for i in range(len(label_names))
        ]  # Create category ID mapping.
        annotations = []

        label_paths_dict = self._fetch_bitmask_paths(raster_dir, maps)
        for map_name, label_paths in tqdm(
            label_paths_dict.items(), desc="Processing bitmasks"
        ):
            filtered_label_paths = set(
                [i for i in label_paths if any(label in i for label in label_aliases)]
            )  # Check any bitmask that contains a label alias.
            detected_label_names = []

            for i in filtered_label_paths:
                for alias in label_aliases:
                    if alias in i:
                        detected_label_names.append(bitmask_reverse_index[alias])

            all_bboxes = [
                self.construct_bounding_boxes(i, bbox_shape)
                for i in filtered_label_paths
            ]
            ids = [filename_to_id(i) for i in filtered_label_paths]
            image_id = filename_to_id(os.path.join(raster_dir, map_name + ".tif"))

            for bboxes, id, label_name in zip(all_bboxes, ids, detected_label_names):
                category_id = next(
                    (cat["id"] for cat in categories if cat["name"] == label_name), None
                )
                if category_id is not None:
                    annotations.append(
                        {
                            "id": int(id),
                            "image_id": int(image_id),
                            "category_id": int(category_id),
                            "segmentation": [],
                            "area": None,
                            "boxes": bboxes,
                        }
                    )
                else:
                    raise RuntimeError(
                        f"Category id not found for label name: {label_name}"
                    )

        self.coco_dict.update(
            {
                "images": map_metadata,
                "annotations": annotations,
                "categories": categories,
            }
        )

        self.coco_dict = filter_coco_images(
            self.coco_dict
        )  # Filter out images which have no annotations.
        return self.coco_dict
