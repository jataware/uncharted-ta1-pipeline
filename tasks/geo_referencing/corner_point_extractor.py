import logging

from shapely import LineString, Point, Polygon

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import Coordinate

from typing import Dict, Tuple

logger = logging.getLogger("corner_point_extractor")

# multiplier for the length of centerlines from lat/lon labels
# controls the effective search distance for map corners
CENTERLINE_LEN_MULTIPLIER = 2.0


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

        lon_pts: Dict[Tuple[float, float], Coordinate] = input.get_data("lons", {})
        lat_pts: Dict[Tuple[float, float], Coordinate] = input.get_data("lats", {})

        lon_pts_modified = []
        lat_pts_modified = []

        lon_pts_out: Dict[Tuple[float, float], Coordinate] = {}
        lat_pts_out: Dict[Tuple[float, float], Coordinate] = {}

        # loop over lon coords...
        num_corners = 0
        for i_lon, (lon_key, lon_coord) in enumerate(lon_pts.items()):

            try:
                if i_lon in lon_pts_modified:
                    # this longitude pt has already been fine-tuned; skip
                    continue

                # generate a vertical segment through the center of the lon label
                lon_label_poly = Polygon([(i.x, i.y) for i in lon_coord.get_bounds()])
                lon_label_width = CENTERLINE_LEN_MULTIPLIER * (
                    lon_label_poly.bounds[2] - lon_label_poly.bounds[0]
                )
                # get pixel centre xy for this longitude extraction
                lon_center_x, lon_center_y = lon_coord.get_pixel_alignment()
                lon_line = LineString(
                    [
                        (
                            lon_center_x,
                            lon_center_y + lon_label_width,
                        ),
                        (lon_center_x, lon_center_y - lon_label_width),
                    ]
                )
                # inner loop over lat coords...
                for i_lat, (lat_key, lat_coord) in enumerate(lat_pts.items()):
                    try:
                        if i_lat in lat_pts_modified:
                            # this latitude pt has already been fine-tuned; skip
                            continue

                        # generate a horizontal segment through the center of the lat label
                        lat_label_poly = Polygon(
                            [(i.x, i.y) for i in lat_coord.get_bounds()]
                        )
                        lat_label_width = CENTERLINE_LEN_MULTIPLIER * (
                            lat_label_poly.bounds[2] - lat_label_poly.bounds[0]
                        )
                        # get pixel centre xy for this latitude extraction
                        lat_center_x, lat_center_y = lat_coord.get_pixel_alignment()
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
                            # corner-point detected!
                            num_corners += 1
                            # fine-tune the pixel locations for the corresponding lat and lon pts
                            # and save results
                            lon_key_mod = (lon_key[0], intersection.x)
                            lon_coord.set_pixel_alignment(
                                (intersection.x, intersection.y)
                            )
                            lon_coord._is_corner = True
                            lon_pts_out[lon_key_mod] = lon_coord
                            lon_pts_modified.append(i_lon)

                            lat_key_mod = (lat_key[0], intersection.y)
                            lat_coord.set_pixel_alignment(
                                (intersection.x, intersection.y)
                            )
                            lat_coord._is_corner = True
                            lat_pts_out[lat_key_mod] = lat_coord
                            lat_pts_modified.append(i_lat)

                    except Exception as e:
                        logger.warning(
                            f"Exception analyzing latitude coords: {repr(e)}"
                        )
                        continue

            except Exception as e:
                logger.warning(f"Exception analyzing longitude coords: {repr(e)}")
                continue

        logger.info(f"Number of corner points detected: {num_corners}")
        # finished corner-point detection,
        # if < 4 corners, then append any remaining unmodified lat and lon pts to the output dicts
        if num_corners < 4:
            for i_lon, (lon_key, lon_coord) in enumerate(lon_pts.items()):

                if i_lon not in lon_pts_modified and lon_key not in lon_pts_out:
                    lon_pts_out[lon_key] = lon_coord

            for i_lat, (lat_key, lat_coord) in enumerate(lat_pts.items()):

                if i_lat not in lat_pts_modified and lat_key not in lat_pts_out:
                    lat_pts_out[lat_key] = lat_coord

        # save lons and lats task results
        result = self._create_result(input)
        result.output["lons"] = lon_pts_out
        result.output["lats"] = lat_pts_out
        return result
