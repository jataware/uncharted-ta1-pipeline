import logging
from copy import deepcopy
from shapely import Polygon, box

from tasks.geo_referencing.entities import Coordinate, CoordSource, CoordType
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import MapROI, ROI_MAP_OUTPUT_KEY

from typing import Dict, Tuple

logger = logging.getLogger("coordinates_filter")

NAIVE_FILTER_MINIMUM = 10


class FilterCoordinates(Task):

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        # get coordinates so far
        lon_pts = input.get_data("lons")
        lat_pts = input.get_data("lats")

        # filter the coordinates to retain only those that are deemed valid
        lon_pts_filtered, lat_pts_filtered = self._filter(input, lon_pts, lat_pts)

        logger.info(
            f"Num coordinates after filtering: {len(lat_pts_filtered)} latitude and {len(lon_pts_filtered)}"
        )

        # update the coordinates list
        return self._create_result(input, lon_pts_filtered, lat_pts_filtered)

    def _create_result(
        self,
        input: TaskInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return lon_coords, lat_coords


class UTMStatePlaneFilter(FilterCoordinates):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:

        # get the count and confidence of state plane and utm coordinates
        lon_count_sp, lon_conf_sp = self._get_score(lon_coords, CoordSource.STATE_PLANE)
        lon_count_utm, lon_conf_utm = self._get_score(lon_coords, CoordSource.UTM)
        lat_count_sp, lat_conf_sp = self._get_score(lat_coords, CoordSource.STATE_PLANE)
        lat_count_utm, lat_conf_utm = self._get_score(lat_coords, CoordSource.UTM)

        # if no utm or no state plane coordinates exist then nothing to filter
        if lon_count_sp + lat_count_sp == 0:
            return lon_coords, lat_coords
        if lon_count_utm + lat_count_utm == 0:
            return lon_coords, lat_coords

        # if one has coordinates in both directions while the other doesnt then keep that one
        source_filter = None
        if (
            min(lon_count_utm, lat_count_utm) > 0
            and min(lon_count_sp, lat_count_sp) == 0
        ):
            logger.debug("removing state plane coordinates since one axis has none")
            source_filter = CoordSource.STATE_PLANE
        elif (
            min(lon_count_utm, lat_count_utm) == 0
            and min(lon_count_sp, lat_count_sp) > 0
        ):
            logger.debug("removing utm coordinates since one axis has none")
            source_filter = CoordSource.UTM

        # if still unsure then retain the one with the highest confidence
        # by this point both utm and state plane have coordinates in one or two directions
        if not source_filter:
            source_filter = CoordSource.UTM
            if max(lon_conf_utm, lat_conf_utm) > max(lon_conf_sp, lat_conf_sp):
                logger.debug(
                    "removing state plane coordinates since utm coordinates have higher confidence"
                )
                source_filter = CoordSource.STATE_PLANE
            else:
                logger.debug(
                    "removing utm coordinates since state plane coordinates have higher confidence"
                )

        logger.debug(f"filtering {source_filter} latitude and longitude coordinates")

        return self._filter_source(source_filter, lon_coords), self._filter_source(
            source_filter, lat_coords
        )

    def _get_score(
        self, coords: Dict[Tuple[float, float], Coordinate], source: CoordSource
    ) -> Tuple[int, float]:
        conf = -1
        count = 0
        for _, c in coords.items():
            src = c.get_source()
            if src == source:
                count = count + 1
                if conf < c.get_confidence():
                    conf = c.get_confidence()
        return (count, conf)

    def _filter_source(
        self, source: CoordSource, coords: Dict[Tuple[float, float], Coordinate]
    ) -> Dict[Tuple[float, float], Coordinate]:
        coords_filtered = {}
        for k, c in coords.items():
            if not c.get_source() == source:
                coords_filtered[k] = c
        return coords_filtered


class ROIFilter(FilterCoordinates):
    """
    Coordinate filtering based on Map region-of-interest (ROI)
    """

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _filter(
        self,
        input: TaskInput,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:

        if not ROI_MAP_OUTPUT_KEY in input.data:
            logger.warning("No ROI info available; skipping ROIFilter")
            return (lon_coords, lat_coords)

        map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])

        # use the ROI buffering result to create a "ring" polygon shape around the map's boundaries
        try:
            roi_poly = Polygon(shell=map_roi.buffer_outer, holes=[map_roi.buffer_inner])
        except Exception as ex:
            logger.warning(
                "Exception using inner and outer ROI buffering; just using outer buffering"
            )
            roi_poly = Polygon(shell=map_roi.buffer_outer)

        lon_inputs = deepcopy(lon_coords)  # TODO: is deepcopy necessary?
        lat_inputs = deepcopy(lat_coords)
        lon_counts_initial, lat_counts_initial = self._get_distinct_degrees(
            lon_inputs, lat_inputs
        )

        # --- do ROI filtering
        lons, lats = self._filter_roi(lon_inputs, lat_inputs, roi_poly)
        lon_counts, lat_counts = self._get_distinct_degrees(lons, lats)

        # --- adjust filtering based on distance to roi if insufficient points
        if lon_counts < 2 and lon_counts < lon_counts_initial:
            logger.debug(
                f"only {lon_counts} lon coords after roi filtering so re-adding coordinates"
            )
            lons = self._adjust_filter(lons, lon_coords, roi_poly)

        if lat_counts < 2 and lat_counts < lat_counts_initial:
            logger.debug(
                f"only {lat_counts} lat coords after roi filtering so re-adding coordinates"
            )
            lats = self._adjust_filter(lats, lat_coords, roi_poly)

        # ---
        # TODO: this should be re-factored/merged with other similar geo-ref tasks
        lons, lats = self._validate_lonlat_extractions(lons, lats, input.image.size)

        return lons, lats

    def _adjust_filter(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        coords_all: Dict[Tuple[float, float], Coordinate],
        roi_poly: Polygon,
    ) -> Dict[Tuple[float, float], Coordinate]:
        """
        Adjust ROI filtering based on the rectangular ROI if insufficient points
        """
        distinct_degs = set(map(lambda x: x[1].get_parsed_degree(), coords.items()))
        # create bbox of the ROI polygon
        roi_bbox = box(*roi_poly.bounds)

        coords_to_add = {}
        for (deg, i), coord in coords_all.items():
            if (deg, i) in coords:
                # this coord is already in valid set
                continue
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_bbox):
                # save as coord to keep
                coords_to_add[(deg, i)] = coord

        # sort by confidence
        coords_to_add = sorted(
            coords_to_add.items(), key=lambda x: x[1].get_confidence(), reverse=True
        )
        # add more coordinate results
        for (deg, i), coord in coords_to_add:
            if len(distinct_degs) >= 2:
                break
            # TODO: flag this coord as being outside the map's ROI
            coord._confidence *= 0.5  # re-add coord with reduced confidence
            coords[(deg, i)] = coord
            distinct_degs.add(deg)
            logger.debug(f"re-adding coordinate: {deg} ({coord.get_pixel_alignment()})")

        return coords

    def _get_distinct_degrees(
        self,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[int, int]:
        """
        Get the number of unique degree values for extracted lat and lon values
        """
        lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lat_coords.items()))
        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lon_coords.items()))
        return len(lons_distinct), len(lats_distinct)

    def _validate_lonlat_extractions(
        self,
        lon_results: Dict[Tuple[float, float], Coordinate],
        lat_results: Dict[Tuple[float, float], Coordinate],
        im_size: Tuple[float, float],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Add an inferred anchor lat/lon pt. if needed
        """
        # TODO: this should be re-factored/merged with other similar geo-ref tasks

        num_lat_pts = len(lat_results)
        num_lon_pts = len(lon_results)
        logger.debug(
            f"point count after exclusion lat,lon: {num_lat_pts},{num_lon_pts}"
        )

        # check number of unique lat and lon values
        num_lat_pts = len(set([x[0] for x in lat_results]))
        num_lon_pts = len(set([x[0] for x in lon_results]))
        logger.debug(f"distinct after roi lat,lon: {num_lat_pts},{num_lon_pts}")

        if num_lon_pts >= 2 and num_lat_pts == 1:
            # estimate additional lat pt (based on lon pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[1]) for k, v in lon_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lat_pt = list(lat_results.items())[0]
                # new_y = im_size[1]-1
                new_y = 0 if lat_pt[0][1] > im_size[1] / 2 else im_size[1] - 1
                new_lat = -deg_per_pxl * (new_y - lat_pt[0][1]) + lat_pt[0][0]
                coord = Coordinate(
                    CoordType.DERIVED_KEYPOINT,
                    "",
                    new_lat,
                    CoordSource.LAT_LON,
                    True,
                    pixel_alignment=(lat_pt[1].to_deg_result()[1], new_y),
                    confidence=0.6,
                )
                lat_results[(new_lat, new_y)] = coord

        elif num_lat_pts >= 2 and num_lon_pts == 1:
            # estimate additional lon pt (based on lat pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[0]) for k, v in lat_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lon_pt = list(lon_results.items())[0]
                # new_x = im_size[0]-1
                new_x = 0 if lon_pt[0][1] > im_size[0] / 2 else im_size[0] - 1
                new_lon = -deg_per_pxl * (new_x - lon_pt[0][1]) + lon_pt[0][0]
                coord = Coordinate(
                    CoordType.DERIVED_KEYPOINT,
                    "",
                    new_lon,
                    CoordSource.LAT_LON,
                    False,
                    pixel_alignment=(new_x, lon_pt[1].to_deg_result()[1]),
                    confidence=0.6,
                )
                lon_results[(new_lon, new_x)] = coord

        return (lon_results, lat_results)

    def _filter_roi(
        self,
        lon_coords: Dict[Tuple[float, float], Coordinate],
        lat_coords: Dict[Tuple[float, float], Coordinate],
        roi_poly: Polygon,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Filter extracted coordinates based on the map ROI
        """

        if not roi_poly:
            return (lon_coords, lat_coords)

        lon_out = {}
        lat_out = {}
        for (deg, y), coord in list(lat_coords.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this latitude pt
                lat_out[(deg, y)] = coord
            else:
                logger.debug(
                    f"removing out-of-bounds latitude point: {deg} ({coord.get_pixel_alignment()})"
                )
        for (deg, x), coord in list(lon_coords.items()):
            coord_poly = Polygon([(pt.x, pt.y) for pt in coord.get_bounds()])
            if coord_poly.intersects(roi_poly):
                # keep this longitude pt
                lon_out[(deg, x)] = coord
            else:
                logger.debug(
                    f"removing out-of-bounds longitude point: {deg} ({coord.get_pixel_alignment()})"
                )

        return (lon_out, lat_out)
