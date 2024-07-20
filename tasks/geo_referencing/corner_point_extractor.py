from curses import meta
import logging
from unittest.mock import DEFAULT

from shapely import LineString, Point

from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    MetadataExtractionOutput,
)
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    Coordinate,
    GroundControlPoint,
    CORNER_POINTS_OUTPUT_KEY,
)

from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("corner_point_extractor")


class CornerPointExtractor(Task):
    """
    Extracts corner points from longitude and latitude coordinates.

    This task takes in longitude and latitude coordinates and generates corner points
    by finding the intersection of vertical and horizontal lines passing through the
    center of each coordinate label.

    Args:
        task_id (str): The ID of the task.

    Attributes:
        task_id (str): The ID of the task.

    """

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        """
        Runs the corner point extraction task.

        Args:
            input (TaskInput): The input data for the task.

        Returns:
            TaskResult: The result object updated with the corner points if present.

        """
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
                # generate a horizontal segment through the center of the lat label
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
                        (Point(lon_key[0], lat_key[0]), intersection)
                    )

        if len(intersection_points) >= 3:
            logger.info(f"Found {len(intersection_points)} corner points")

            output: List[GroundControlPoint] = []
            # write out as gcps
            for i, point in enumerate(intersection_points):
                geo_point = point[0]
                pixel_point = point[1]
                gcp = GroundControlPoint(
                    id=f"corner.{str(i)}",
                    longitude=geo_point.x,
                    latitude=geo_point.y,
                    pixel_x=pixel_point.x,
                    pixel_y=pixel_point.y,
                    confidence=1.0,
                )
                output.append(gcp)
        else:
            logger.info(
                f"Found {len(intersection_points)} corner points, require at least 3.  Corner point referencing not available."
            )
            output = []

        result = self._create_result(input)
        result.output[CORNER_POINTS_OUTPUT_KEY] = output

        return result
