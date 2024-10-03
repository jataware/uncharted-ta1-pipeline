import logging

from random import randint

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import QUERY_POINTS_OUTPUT_KEY
from tasks.geo_referencing.georeference import QueryPoint
from tasks.segmentation.entities import (
    MapSegmentation,
    SEGMENTATION_OUTPUT_KEY,
    SEGMENT_MAP_CLASS,
)
from tasks.segmentation.segmenter_utils import get_segment_bounds

from typing import List, Tuple, Dict
from shapely import Point, Polygon, MultiPolygon, concave_hull
from shapely.ops import nearest_points

GEOCOORD_DIST_THRES = 30
NUM_PTS = 8

logger = logging.getLogger("ground_control_points")


class CreateGroundControlPoints(Task):
    def __init__(self, task_id: str, create_random_pts: bool = True):
        self.create_random_pts = create_random_pts
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:

        # check if query points already defined
        query_pts = None
        if QUERY_POINTS_OUTPUT_KEY in input.request:
            query_pts = input.request[QUERY_POINTS_OUTPUT_KEY]
        if query_pts and len(query_pts) > 0:
            logger.info("ground control points already exist")
            return self._create_result(input)

        # no query points exist, so create them...
        # get map segmentation ROI as a shapely polygon (without any dilation buffering)
        poly_map = []
        if SEGMENTATION_OUTPUT_KEY in input.data:
            segmentation = MapSegmentation.model_validate(
                input.data[SEGMENTATION_OUTPUT_KEY]
            )
            poly_map = get_segment_bounds(segmentation, SEGMENT_MAP_CLASS)

        if self.create_random_pts or not poly_map:
            # create random GCPs...
            query_pts = self._create_random_query_points(input, num_pts=NUM_PTS)
            logger.info(f"created {len(query_pts)} random ground control points")
        else:
            # create GCPs based on geo-coord pixel locations...
            # use 1st (highest ranked) map segment
            poly_map = poly_map[0]
            # get extracted lat and lon coords
            lon_pts = input.get_data("lons", {})
            lat_pts = input.get_data("lats", {})
            num_inner_gcps = max(NUM_PTS - (len(lon_pts) + len(lat_pts)), 0)

            query_pts = self._create_geo_coord_query_points(
                input.raster_id, poly_map, lon_pts, lat_pts
            )
            logger.info(
                f"created {len(query_pts)} geo-coord based ground control points"
            )
            if num_inner_gcps > 0:
                logger.info(
                    f"Also creating {num_inner_gcps} random ground control points"
                )
                query_pts.extend(
                    self._create_random_query_points(input, num_pts=num_inner_gcps)
                )

        # add them to the output
        result = self._create_result(input)
        result.output[QUERY_POINTS_OUTPUT_KEY] = query_pts

        return result

    def _create_random_query_points(
        self, input: TaskInput, num_pts=10
    ) -> List[QueryPoint]:
        """
        create N random ground control points roughly around the middle of the ROI (or failing that the middle of the image)
        """
        min_x = min_y = max_x = max_y = 0
        roi = input.get_data("roi")
        if roi and len(roi) > 0:
            roi_x = list(map(lambda x: x[0], roi))
            roi_y = list(map(lambda x: x[1], roi))

            max_x, max_y = max(roi_x), max(roi_y)
            min_x, min_y = min(roi_x), min(roi_y)
        else:
            max_x, max_y = input.image.size

        coords = self._create_random_coordinates(min_x, min_y, max_x, max_y, n=num_pts)
        return [
            QueryPoint(
                input.raster_id, (c[0], c[1]), None, properties={"label": "random"}
            )
            for c in coords
        ]

    def _create_geo_coord_query_points(
        self, raster_id: str, poly_map: Polygon, lon_pts: Dict, lat_pts: Dict
    ) -> List[QueryPoint]:
        """
        create ground control points at approximately the pixel locations where geo-coordinates have been extracted
        """

        # do polygon of roi
        gcp_pts = []
        # GCPs based on extracted longitude geo-coords
        for coord in lon_pts.values():

            pt = Point(coord.get_pixel_alignment())

            if coord.is_corner():
                # is a corner-pt, no need to adjust GCP location
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
                continue

            if pt.intersects(poly_map):
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
            else:
                # pt doesn't intersect map ROI, adjust y pixel location
                dist1 = poly_map.distance(pt)
                pt_on_map = nearest_points(poly_map, pt)[0]
                pt2 = Point(pt.x, pt_on_map.y)
                dist2 = poly_map.distance(pt2)
                # use adjusted pt if it's closer to map roi
                if dist2 < dist1:
                    pt = pt2
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
        # GCPs based on extracted latitude geo-coords
        for coord in lat_pts.values():

            pt = Point(coord.get_pixel_alignment())

            if coord.is_corner():
                # is a corner-pt, no need to adjust GCP location
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
                continue

            if pt.intersects(poly_map):
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
            else:
                # pt doesn't intersect map ROI, adjust x pixel location
                dist1 = poly_map.distance(pt)
                pt_on_map = nearest_points(poly_map, pt)[0]
                pt2 = Point(pt_on_map.x, pt.y)
                dist2 = poly_map.distance(pt2)
                # use adjusted pt if it's closer to map roi
                if dist2 < dist1:
                    pt = pt2
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)

        return [
            QueryPoint(raster_id, (pt.x, pt.y), None, properties={"label": "geo_coord"})
            for pt in gcp_pts
        ]

    def _create_random_coordinates(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        n: int = 10,
        buffer: float = 0.25,
    ) -> List[Tuple[int, int]]:
        # randomize x & y coordinates fitting between boundaries
        range_x = max_x - min_x
        range_y = max_y - min_y

        min_x_buf = min_x + range_x * buffer
        max_x_buf = max_x - range_x * buffer
        min_y_buf = min_y + range_y * buffer
        max_y_buf = max_y - range_y * buffer

        return [
            (
                randint(int(min_x_buf), int(max_x_buf)),
                randint(int(min_y_buf), int(max_y_buf)),
            )
            for _ in range(n)
        ]

    def _do_pts_overlap(
        self,
        input_pt: Point,
        ref_pts: List[Point],
        dist_thres: int = GEOCOORD_DIST_THRES,
    ) -> bool:
        """
        Check if an input point overlaps with any of list of reference points, within a distance threshold
        """
        for pt in ref_pts:
            if input_pt.dwithin(pt, distance=dist_thres):
                return True
        return False
