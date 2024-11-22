import logging
import numpy as np
from typing import Dict, Tuple
from tasks.common.task import Task, TaskInput, TaskResult
from shapely import Polygon
from tasks.geo_referencing.entities import (
    ROI_MAP_OUTPUT_KEY,
    MapROI,
    Coordinate,
    CoordSource,
    CoordType,
)
from tasks.geo_referencing.util import get_distinct_degrees
from tasks.common.task import Task, TaskInput, TaskResult

logger = logging.getLogger("finalize_coordinates")

COLINEARITY_THRES = 0.05  # co-linearity threshold (percentage)


class FinalizeCoordinates(Task):
    """
    Finalize coordinate extractions.
    Includes checking for co-linear or ill-conditioned coord spacing
    """

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """

        # get the extracted coordinates
        lon_pts = input.get_data("lons", {})
        lat_pts = input.get_data("lats", {})

        if len(lon_pts) == 0 and len(lat_pts) == 0:
            # No coords available; skip this task
            return self._create_result(input)

        # get number of corners (only need to iterate over lats or lons, since a corner includes a lat/lon pair)
        num_corners = sum(
            [1 if x.get_type() == CoordType.CORNER else 0 for x in lon_pts.values()]
        )
        if num_corners >= 3:
            # 3 or more corner points found; assume coords spacing is adequate and skip this task
            return self._create_result(input)

        # get map-roi result
        map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])
        if not map_roi:
            # No map ROI available; skip this task
            return self._create_result(input)
        map_poly = Polygon(map_roi.map_bounds)

        # check if all coords are too co-linear, and if so, infer an additional anchor coord
        lon_pts = self._check_colinearity(lon_pts, map_poly, input.image.size)
        lat_pts = self._check_colinearity(lat_pts, map_poly, input.image.size)

        # an additional check to infer a 3rd anchor point, if needed
        lon_pts = self._infer_third_coord(lon_pts, map_poly, input.image.size)
        lat_pts = self._infer_third_coord(lat_pts, map_poly, input.image.size)

        # update the coordinates lists
        result = self._create_result(input)
        result.output["lons"] = lon_pts
        result.output["lats"] = lat_pts
        return result

    def _check_colinearity(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        map_poly: Polygon,
        im_size_wh: Tuple[int, int],
    ) -> Dict[Tuple[float, float], Coordinate]:
        """
        Check if pixel spacing of the extracted coordinates is too colinear,
        and if so, add an additional derived coord
        (to prevent ill-conditioned polynomial regression results for the georeferencing transform)
        """

        num_distinct_degs = get_distinct_degrees(coords)
        if num_distinct_degs < 2:
            return coords

        # notes for coord formatting
        # lon = (deg, x) y
        # lat = (deg, y) x
        # i == major axis (ie x-axis for lon, y for lat)
        # j == minor axis

        i_idx = 0  # lon, x = major axis
        j_idx = 1
        is_lat = next(iter(coords.values())).is_lat()
        if is_lat:
            i_idx = 1  # lat, y = major axis
            j_idx = 0

        i_vals = [c.get_pixel_alignment()[i_idx] for c in coords.values()]
        i_range = abs(max(i_vals) - min(i_vals))
        j_vals = [c.get_pixel_alignment()[j_idx] for c in coords.values()]
        j_range = abs(max(j_vals) - min(j_vals))

        skew_slope = 0.0
        if i_range > 0:
            skew_slope = j_range / i_range

        if skew_slope < COLINEARITY_THRES:
            # extracted coords are co-linear (approx aligned along i-axis)
            # so assume "minimal" rotation/skew of map, and add an extra keypoint to help 'anchor'
            # the polynomial regression fit (prevent erratic mapping behaviour)

            # estimate skew (rotation) of map wrt i-axis
            m, b = np.polyfit(i_vals, j_vals, 1)
            # get the first keypoint
            c = next(iter(coords.values()))
            deg_pt = c.get_parsed_degree()
            pxl_i = c.get_pixel_alignment()[i_idx]
            pxl_j = c.get_pixel_alignment()[j_idx]
            # mid point of map ROI in j-axis
            map_j_mid = (map_poly.bounds[0 + j_idx] + map_poly.bounds[2 + j_idx]) / 2
            # new j value (far from the others)
            new_j = (
                map_poly.bounds[0 + j_idx]
                if pxl_j > map_j_mid
                else map_poly.bounds[2 + j_idx]
            )
            i_offset = int(m * (pxl_j - new_j))
            if i_offset == 0:
                i_offset = 1  # jitter by 1 pixel (at least) -- improves geo-transform stability
            new_i = max(min(pxl_i + i_offset, im_size_wh[i_idx] - 1), 0)
            logger.info(
                "Adding an anchor keypoint (assuming minimal skew): deg: {}, i,j: {},{}".format(
                    deg_pt, new_i, new_j
                )
            )
            new_xy = (new_i, new_j) if i_idx == 0 else (new_j, new_i)
            # save inferred coordinate
            new_coord = Coordinate(
                CoordType.DERIVED_KEYPOINT,
                "",
                deg_pt,
                CoordSource.INFERENCE,
                is_lat,
                pixel_alignment=new_xy,
                confidence=0.5,
            )
            coords[(deg_pt, new_i)] = new_coord

        return coords

    def _infer_third_coord(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        map_poly: Polygon,
        im_size_wh: Tuple[int, int],
    ) -> Dict[Tuple[float, float], Coordinate]:
        """
        Check if a 3rd coordinate needs still needs to be inferred, to help "anchor" the
        polynomial regression results for the georeferencing transform
        (eg such as if 2 coords have been extracted in opposite diagonal corners of a map)
        """

        num_distinct_degs = get_distinct_degrees(coords)
        if num_distinct_degs != 2 or len(coords) != 2:
            return coords

        # there are only 2 unique keypoints; not enough to reliably handle rotation in geo-projection,
        # so assume no rotation/skew of map, and add a 3rd keypoint to help 'anchor'
        # the polynomial regression fit (prevent erratic mapping behaviour)
        i_idx = 0  # lon, x = major axis
        j_idx = 1
        is_lat = next(iter(coords.values())).is_lat()
        if is_lat:
            i_idx = 1  # lat, y = major axis
            j_idx = 0

        # get the first keypoint
        c = next(iter(coords.values()))
        deg_pt = c.get_parsed_degree()
        pxl_i = c.get_pixel_alignment()[i_idx]
        pxl_j = c.get_pixel_alignment()[j_idx]

        # mid point of map ROI in j-axis
        map_j_mid = (map_poly.bounds[0 + j_idx] + map_poly.bounds[2 + j_idx]) / 2
        # new j value (far from the first coord)
        new_j = (
            map_poly.bounds[0 + j_idx]
            if pxl_j > map_j_mid
            else map_poly.bounds[2 + j_idx]
        )
        new_i = pxl_i + 1  # jitter by 1 pixel -- improves geo-transform stability
        new_i = max(min(new_i, im_size_wh[i_idx] - 1), 0)
        logger.info(
            "Adding a 3rd anchor keypoint (assuming no skew): deg: {}, i,j: {},{}".format(
                deg_pt, new_i, new_j
            )
        )
        new_xy = (new_i, new_j) if i_idx == 0 else (new_j, new_i)
        # save inferred coordinate
        new_coord = Coordinate(
            CoordType.DERIVED_KEYPOINT,
            "",
            deg_pt,
            CoordSource.INFERENCE,
            is_lat,
            pixel_alignment=new_xy,
            confidence=0.5,
        )
        coords[(deg_pt, new_i)] = new_coord

        return coords
