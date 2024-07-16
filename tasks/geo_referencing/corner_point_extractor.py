import json
import logging
import pprint

from shapely import LineString, Point

from schema.cdr_schemas.georeference import Geom_Point, GroundControlPoint, Pixel_Point
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import Coordinate

from typing import Dict, List, Tuple

logger = logging.getLogger("corner_point_extractor")


class CornerPointExtractor(Task):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        logger.info(
            f"==========> running corner point extractor task index {input.task_index} with id {self._task_id}"
        )

        # check if query points already defined
        corner_points = None

        lon_pts: Dict[Tuple[float, float], Coordinate] = input.get_data("lons")
        lat_pts: Dict[Tuple[float, float], Coordinate] = input.get_data("lats")

        intersection_points: List[Tuple[Point, Point]] = []

        for lon_key, lon_coord in lon_pts.items():
            # generate a vertical segment through the center of the lon label
            lon_bounds = lon_coord.get_bounds()
            lon_label_width = lon_bounds[1].x - lon_bounds[0].x
            lon_center_x = lon_bounds[0].x + (lon_bounds[1].x - lon_bounds[0].x) / 2.0
            lon_center_y = lon_bounds[0].y + (lon_bounds[2].y - lon_bounds[0].y) / 2.0
            lon_line = LineString(
                [
                    (
                        lon_center_x,
                        lon_center_y + lon_label_width,
                    ),
                    (lon_center_x, lon_center_y - lon_label_width),
                ]
            )
            for lat_key, lat_coord in lat_pts.items():
                # generate a vertical segment through the center of the lat label
                lat_bounds = lat_coord.get_bounds()
                lat_label_width = lat_bounds[1].x - lat_bounds[0].x
                lat_center_x = (
                    lat_bounds[0].x + (lat_bounds[1].x - lat_bounds[0].x) / 2.0
                )
                lat_center_y = (
                    lat_bounds[0].y + (lat_bounds[2].y - lat_bounds[0].y) / 2.0
                )
                lat_line = LineString(
                    [
                        (
                            lat_center_x + lat_label_width,
                            lat_center_y,
                        ),
                        (lat_center_x - lat_label_width, lat_center_y),
                    ]
                )
                # find the intersection of the two lines
                intersection = lon_line.intersection(lat_line)
                if isinstance(intersection, Point):
                    intersection_points.append(
                        (Point(-lon_key[0], lat_key[0]), intersection)
                    )

        if len(intersection_points) == 4:
            logger.info("Found 4 intersection points")
            output = {"gcps": []}
            # write out as gcps
            for i, point in enumerate(intersection_points):
                geo_point = point[0]
                pixel_point = point[1]
                gcp = GroundControlPoint(
                    gcp_id=str(i),
                    map_geom=Geom_Point(longitude=geo_point.x, latitude=geo_point.y),
                    px_geom=Pixel_Point(
                        columns_from_left=pixel_point.x, rows_from_top=pixel_point.y
                    ),
                    model="corner_point_extractor",
                    model_version="0.0.1",
                    crs="EPSG:4267",
                )
                output["gcps"].append(gcp.model_dump())

            # convert output to json
            with open("corner_points.json", "w") as outfile:
                json.dump(output, outfile, indent=4)

        # if we have intersection points, we can use them as corner points
        # add them to the output
        result = self._create_result(input)
        result.output["corner_points"] = "scaramouche"

        return result
