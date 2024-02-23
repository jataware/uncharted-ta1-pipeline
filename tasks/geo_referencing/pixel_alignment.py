import logging
import uuid

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import Coordinate
from tasks.geo_referencing.util import ocr_to_coordinates
from util.json import read_json_file

from typing import Any, Dict, Tuple

logger = logging.getLogger("pixel_alignment")


class TickDetector(Task):
    _coordinates_file: str

    def __init__(self, task_id: str, coordinates_file: str):
        super().__init__(task_id)
        self._coordinates_file = coordinates_file

    def run(self, input: TaskInput) -> TaskResult:
        logger.info(f"running tick detector with id {self._task_id}")

        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # read exact pixels from file
        coordinates_actual = read_json_file(self._coordinates_file)
        print(coordinates_actual)
        for c_f in coordinates_actual:
            if c_f["map_id"] == input.raster_id:
                logger.info(f"map {input.raster_id} found in pixel alignment resource")
                lon_pts = self._map_pixels(input, lon_pts, c_f["coordinates"]["lons"])
                lat_pts = self._map_pixels(input, lat_pts, c_f["coordinates"]["lats"])

        result = super()._create_result(input)
        result.output["lons"] = lon_pts
        result.output["lats"] = lat_pts
        return result

    def _map_pixels(
        self,
        input: TaskInput,
        coordinates: Dict[Tuple[float, float], Coordinate],
        coordinates_actual: Dict[str, Any],
    ) -> Dict[Tuple[float, float], Coordinate]:
        # cycle through each parsed coordinate to fix the alignment if specified
        # need to recreate coordinate dictionary as pixels may have changed
        output = {}
        for _, c in coordinates.items():
            deg_str = f"{c.get_parsed_degree()}"
            if deg_str in coordinates_actual:
                logger.info(
                    f"checking for alignment correction since {deg_str} found in corrected alignment list"
                )
                for ca in coordinates_actual[deg_str]:
                    ca_x = ca["x"]
                    ca_y = ca["y"]

                    # coordinate is a match if the current pixel alignment is roughly aligned with the uncorrected coordinate
                    c_x, c_y = c.get_pixel_alignment()
                    if abs(c_x - ca_x) < 10 and abs(c_y - ca_y) < 10:
                        new_alignment = (ca["x_actual"], ca["y_actual"])
                        logger.info(
                            f"adjusting pixel from {(c_x, c_y)} to {new_alignment}"
                        )
                        # adjust the pixel alignment
                        c.set_pixel_alignment(new_alignment)

                        # add the adjusted coordinate to the params
                        self._add_param(
                            input,
                            str(uuid.uuid4()),
                            f"coordinate-{c.get_type()}-{c.get_derivation()}",
                            {
                                "bounds": ocr_to_coordinates(c.get_bounds()),
                                "text": c.get_text(),
                                "parsed": c.get_parsed_degree(),
                                "type": "latitude" if c.is_lat() else "longitude",
                                "pixel_alignment": c.get_pixel_alignment(),
                                "confidence": c.get_confidence(),
                            },
                            "extracted aligned coordinate",
                        )
            output[c.to_deg_result()[0]] = c

        return output
