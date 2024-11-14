import logging
from shapely import Polygon, Point, box
from typing import Dict, List, Tuple
from tasks.geo_referencing.entities import CoordStatus, CoordSource, Coordinate
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import MapROI, ROI_MAP_OUTPUT_KEY
from tasks.geo_referencing.util import get_distinct_degrees, is_coord_from_source

logger = logging.getLogger("roi_filter")

DISTINCT_DEGREES_MIN = 2
DIST_PERCENT_THRES = 16


class ROIFilter(Task):
    """
    Coordinate filtering based on Map region-of-interest (ROI)
    """

    def __init__(self, task_id: str, coord_source_check: List[CoordSource] = []):
        self._coord_source_check = coord_source_check
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """

        if not ROI_MAP_OUTPUT_KEY in input.data:
            logger.warning("No ROI info available; skipping ROIFilter")
            return self._create_result(input)

        map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])

        # use the ROI buffering result to create a "ring" polygon shape around the map's boundaries
        try:
            roi_poly = Polygon(shell=map_roi.buffer_outer, holes=[map_roi.buffer_inner])
        except Exception as ex:
            logger.warning(
                "Exception using inner and outer ROI buffering; just using outer buffering"
            )
            roi_poly = Polygon(shell=map_roi.buffer_outer)

        # get coordinates so far
        lons = input.get_data("lons", {})
        lats = input.get_data("lats", {})
        lons_excluded = input.get_data("lons_excluded", {})
        lats_excluded = input.get_data("lats_excluded", {})
        if self._coord_source_check:
            if not is_coord_from_source(
                lons, self._coord_source_check
            ) and not is_coord_from_source(lats, self._coord_source_check):
                # No coordinates are from extractors {self._coord_source_check}; skipping ROI filtering
                return self._create_result(input)

        # check if each lat and lon is within the desired ROI
        lons_ok = {}
        lons_excluded = {}
        lats_ok = {}
        lats_excluded = {}
        for (deg, y), coord in list(lats.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this latitude pt
                lats_ok[(deg, y)] = coord
            else:
                coord._status = CoordStatus.OUTSIDE_MAP_ROI
                lats_excluded[(deg, y)] = coord
                logger.debug(
                    f"removing out-of-bounds latitude point: {deg} ({coord.get_pixel_alignment()})"
                )
        for (deg, x), coord in list(lons.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this longitude pt
                lons_ok[(deg, x)] = coord
            else:
                coord._status = CoordStatus.OUTSIDE_MAP_ROI
                lons_excluded[(deg, x)] = coord
                logger.debug(
                    f"removing out-of-bounds longitude point: {deg} ({coord.get_pixel_alignment()})"
                )

        lon_counts = get_distinct_degrees(lons_ok)
        lat_counts = get_distinct_degrees(lats_ok)

        # --- adjust filtering based on distance to roi if insufficient points
        if lon_counts < DISTINCT_DEGREES_MIN and len(lons_excluded) > 0:
            logger.debug(
                f"Only {lon_counts} lon coords after roi filtering so re-adding coordinates"
            )
            lons_ok, lons_excluded = self._adjust_filter(
                lons_ok, lons_excluded, Polygon(map_roi.map_bounds)
            )

        if lat_counts < DISTINCT_DEGREES_MIN and len(lats_excluded) > 0:
            logger.debug(
                f"Only {lat_counts} lat coords after roi filtering so re-adding coordinates"
            )
            lats_ok, lats_excluded = self._adjust_filter(
                lats_ok, lats_excluded, Polygon(map_roi.map_bounds)
            )

        logger.info(
            f"Num coordinates after ROI filtering: {len(lats_ok)} latitudes and {len(lons_ok)} longitudes"
        )
        # update the coordinates lists
        result = self._create_result(input)
        result.output["lons"] = lons_ok
        result.output["lats"] = lats_ok
        result.output["lons_excluded"] = lons_excluded
        result.output["lats_excluded"] = lats_excluded

        return result

    def _adjust_filter(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        coords_excluded: Dict[Tuple[float, float], Coordinate],
        roi_poly: Polygon,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Adjust ROI filtering based on the distance from coords to map region
        """
        distinct_degs = set(map(lambda x: x[1].get_parsed_degree(), coords.items()))
        # create bbox of the ROI polygon
        roi_bbox = box(*roi_poly.bounds)
        # get distance normalization factor (100 / min of map width or height)
        dist_percent_factor = max(
            (roi_bbox.bounds[2] - roi_bbox.bounds[0]),
            (roi_bbox.bounds[3] - roi_bbox.bounds[1]),
        )
        dist_percent_factor = 100.0 / dist_percent_factor

        coords_to_add = {}  # coord key -> adjusted distance
        for (deg, i), coord in coords_excluded.items():
            if (deg, i) in coords:
                # this coord is already in valid set
                continue
            # calc the coord-to-map distance (as a percentage of map pixel size)
            dist = abs(roi_bbox.distance(Point(coord.get_pixel_alignment())))
            dist = max(dist, 1.0) * dist_percent_factor
            if dist < DIST_PERCENT_THRES:
                # store coord distance to the map roi, adjusted for confidence
                coords_to_add[(deg, i)] = dist * max(
                    (1.0 - coord.get_confidence(), 0.1)
                )

        # sort by acsending distance
        coords_to_add = sorted(coords_to_add.items(), key=lambda x: x[1], reverse=False)
        # add more coordinate results
        for (deg, i), dist in coords_to_add:
            if len(distinct_degs) >= DISTINCT_DEGREES_MIN:
                break
            c = coords_excluded.pop((deg, i))
            c._confidence *= 0.5  # re-add coord with reduced confidence
            coords[(deg, i)] = c
            distinct_degs.add(deg)
            logger.debug(f"Re-adding coordinate: {deg} ({c.get_pixel_alignment()})")

        return coords, coords_excluded
