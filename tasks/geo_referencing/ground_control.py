import logging
import random
from random import randint
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    QUERY_POINTS_OUTPUT_KEY,
    MapROI,
    ROI_MAP_OUTPUT_KEY,
    CoordType,
)
from tasks.geo_referencing.georeference import QueryPoint
from typing import List, Tuple, Dict
from shapely import Point, Polygon, box
from shapely.ops import nearest_points

GEOCOORD_DIST_THRES = 30
MAX_GCPS = 8
MIN_GCPS = 4

logger = logging.getLogger("ground_control_points")


class CreateGroundControlPoints(Task):
    def __init__(self, task_id: str, create_random_pts: bool = True):
        random.seed(911)
        self.create_random_pts = create_random_pts
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:

        # check if query points already defined
        # (eg from legacy AI4CMA contest data input)
        query_pts = None
        if QUERY_POINTS_OUTPUT_KEY in input.request:
            query_pts = input.request[QUERY_POINTS_OUTPUT_KEY]
        if query_pts and len(query_pts) > 0:
            logger.info("Using existing GCPs")
            return self._create_result(input)

        # no query points exist, so create them...
        map_roi = None
        if ROI_MAP_OUTPUT_KEY in input.data:
            # get map ROI bounds (without inner/outer buffering)
            map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])

        if self.create_random_pts or not map_roi:
            # create random GCPs...
            roi_xy = (
                map_roi.map_bounds
                if map_roi
                else [0, 0, input.image.size[0], input.image.size[1]]
            )
            query_pts = self._create_random_ground_control_pts(
                input.raster_id, roi_xy, num_pts=MAX_GCPS
            )
            logger.info(f"Created {len(query_pts)} random GCPs")
        else:
            # create GCPs based on geo-coord pixel locations...
            # get extracted lat and lon coords
            lon_pts = input.get_data("lons", {})
            lat_pts = input.get_data("lats", {})
            logger.info(
                f"Creating GCPs from {len(lon_pts)} longitude and {len(lat_pts)} latitude extracted keypoints"
            )

            query_pts = self._create_geo_coord_ground_control_pts(
                input.raster_id, map_roi, lon_pts, lat_pts
            )
            logger.info(f"Created {len(query_pts)} geo-coord based GCPs")

        # add them to the output
        result = self._create_result(input)
        result.output[QUERY_POINTS_OUTPUT_KEY] = query_pts

        return result

    def _create_random_ground_control_pts(
        self, raster_id: str, roi_bbox: List, num_pts=10
    ) -> List[QueryPoint]:
        """
        Create N random GCPs roughly around the middle of the ROI (or failing that the middle of the image)
        """
        [min_x, min_y, max_x, max_y] = roi_bbox

        coords = self._create_random_coordinates(min_x, min_y, max_x, max_y, n=num_pts)
        return [
            QueryPoint(raster_id, (c[0], c[1]), None, properties={"label": "random"})
            for c in coords
        ]

    def _create_geo_coord_ground_control_pts(
        self, raster_id: str, map_roi: MapROI, lon_pts: Dict, lat_pts: Dict
    ) -> List[QueryPoint]:
        """
        Create GCPs at approximately the pixel locations where geo-coordinates have been extracted
        """
        map_poly = Polygon(map_roi.map_bounds)
        outer_poly = Polygon(map_roi.buffer_outer)

        gcp_pts: List[Point] = []
        # --- do corner pts
        for coord in lon_pts.values():
            if coord.get_type() == CoordType.CORNER:
                pt = Point(coord.get_pixel_alignment())
                # is a corner-pt, no need to adjust GCP location
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)

        for coord in lat_pts.values():
            if coord.get_type() == CoordType.CORNER:
                pt = Point(coord.get_pixel_alignment())
                # is a corner-pt, no need to adjust GCP location
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)

        # --- do remaining non-corner pts
        coords = list(lon_pts.values()) + list(lat_pts.values())
        # shuffle, then sort all coordinates by confidence (shuffling first allows sort 'ties' to be randomized)
        random.shuffle(coords)
        coords.sort(reverse=True, key=lambda c: c.get_confidence())
        for coord in coords:
            if len(gcp_pts) >= MAX_GCPS or coord.get_type() == CoordType.CORNER:
                continue

            pt = Point(coord.get_pixel_alignment())
            if pt.intersects(map_poly):
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)
            else:
                # pt doesn't intersect map ROI, adjust x or y pixel location,
                # if it is lat or lon coord, respectively
                dist1 = map_poly.distance(pt)
                pt_on_map = nearest_points(map_poly, pt)[0]
                pt2 = (
                    Point(pt_on_map.x, pt.y)
                    if coord.is_lat
                    else Point(pt.x, pt_on_map.y)
                )
                dist2 = map_poly.distance(pt2)
                # use adjusted pt if it's closer to map roi
                if dist2 < dist1:
                    pt = pt2
                if not self._do_pts_overlap(pt, gcp_pts):
                    gcp_pts.append(pt)

        gcp_pts = self._check_gcp_map_quadrants(gcp_pts, map_poly, outer_poly)

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

    def _any_pts_in_polygon(self, poly: Polygon, pts: List[Point]) -> bool:
        """
        Check if any points intersect with a polygon
        """
        for pt in pts:
            if pt.intersects(poly):
                return True
        return False

    def _check_gcp_map_quadrants(
        self, gcp_pts: List[Point], map_poly: Polygon, outer_poly: Polygon
    ) -> List[Point]:
        """
        Check if GCP locations exist if all 4 quadrants of a map's area, and create new ones, if needed
        """

        # create quadrant polygons (including ROI with outer "buffering")
        bb_map = map_poly.bounds
        bb_outer = outer_poly.bounds
        bb_map_center = (bb_map[0] + bb_map[2]) / 2, (bb_map[1] + bb_map[3]) / 2
        bb_topleft = box(bb_outer[0], bb_outer[1], bb_map_center[0], bb_map_center[1])
        bb_topright = box(bb_map_center[0], bb_outer[1], bb_outer[2], bb_map_center[1])
        bb_bottemright = box(
            bb_map_center[0], bb_map_center[1], bb_outer[2], bb_outer[3]
        )
        bb_bottemleft = box(
            bb_outer[0], bb_map_center[1], bb_map_center[0], bb_outer[3]
        )

        extra_gcps = []
        if not self._any_pts_in_polygon(bb_topleft, gcp_pts):
            # create a new GCP in the map's top-left
            extra_gcps.append(Point(bb_map[0], bb_map[1]))
        if not self._any_pts_in_polygon(bb_topright, gcp_pts):
            # create a new GCP in the map's top-right
            extra_gcps.append(Point(bb_map[2], bb_map[1]))
        if not self._any_pts_in_polygon(bb_bottemright, gcp_pts):
            # create a new GCP in the map's bottem-right
            extra_gcps.append(Point(bb_map[2], bb_map[3]))
        if not self._any_pts_in_polygon(bb_bottemleft, gcp_pts):
            # create a new GCP in the map's bottem-left
            extra_gcps.append(Point(bb_map[0], bb_map[3]))

        if len(extra_gcps) > 0:
            gcp_pts.extend(extra_gcps)

        return gcp_pts
