import logging
import numpy as np
from typing import Dict, Tuple, Optional
from tasks.common.task import Task, TaskInput, TaskResult
from shapely import Polygon
from tasks.geo_referencing.entities import (
    ROI_MAP_OUTPUT_KEY,
    GEOFENCE_OUTPUT_KEY,
    MAP_SCALE_OUTPUT_KEY,
    MapROI,
    Coordinate,
    CoordSource,
    CoordType,
    DocGeoFence,
    MapScale,
)
from tasks.geo_referencing.scale_analyzer import ScaleAnalyzer, KM_PER_INCH
from tasks.geo_referencing.util import (
    get_distinct_degrees,
    calc_lonlat_slope_signs,
    add_coordinate_to_dict,
)
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

        try:
            # check if all coords are too co-linear, and if so, infer an additional anchor coord
            lon_pts = self._check_colinearity(lon_pts, map_poly, input.image.size)
            lat_pts = self._check_colinearity(lat_pts, map_poly, input.image.size)

            # an additional check to infer a 3rd anchor point, if needed
            lon_pts = self._infer_third_coord(lon_pts, map_poly, input.image.size)
            lat_pts = self._infer_third_coord(lat_pts, map_poly, input.image.size)

        except Exception as e:
            logger.warning(
                f"Exception with finalizing coordinates; discarding any inferred coords - {repr(e)}"
            )

        if min(get_distinct_degrees(lon_pts), get_distinct_degrees(lat_pts)) < 2:
            # still not enough lat/lon keypoints, estimate some "fallback" anchor keypoints...
            logger.info("Estimating additional fallback anchor keypoints")
            geofence: DocGeoFence = input.parse_data(
                GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
            )
            map_scale: MapScale = input.parse_data(
                MAP_SCALE_OUTPUT_KEY, MapScale.model_validate
            )
            lon_pts, lat_pts = self._create_fallback_coords(
                lon_pts, lat_pts, map_poly, geofence, map_scale
            )

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
            coords = add_coordinate_to_dict(new_coord, coords)

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
        coords = add_coordinate_to_dict(new_coord, coords)

        return coords

    def _create_fallback_coords(
        self,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
        map_poly: Polygon,
        geofence: Optional[DocGeoFence],
        map_scale: Optional[MapScale],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Estimate some "fallback" anchor lat and lon keypoints, based on the map scale, map ROI and the geofence
        """

        if not geofence:
            # no geofence available, unable to create fallback coords
            return lons, lats

        try:
            # get lon-lat centerpoint of the geofence
            geofence_lonlat_center = (
                (geofence.geofence.lon_minmax[0] + geofence.geofence.lon_minmax[1]) / 2,
                (geofence.geofence.lat_minmax[0] + geofence.geofence.lat_minmax[1]) / 2,
            )

            # estimate the degrees-per-pixel resolution
            lonlat_per_pixel = self._get_lonlat_per_pixel(
                lons, lats, geofence_lonlat_center, map_scale
            )
            lonlat_slope_signs = calc_lonlat_slope_signs(
                geofence.geofence.lonlat_hemispheres
            )

            # create fallback anchor coords, as needed
            lons = self._create_anchor_keypoints(
                lons,
                geofence_lonlat_center[0],
                lonlat_per_pixel[0] * lonlat_slope_signs[0],
                map_poly,
                is_lat=False,
            )
            lats = self._create_anchor_keypoints(
                lats,
                geofence_lonlat_center[1],
                lonlat_per_pixel[1] * lonlat_slope_signs[1],
                map_poly,
                is_lat=True,
            )

        except Exception as ex:
            logger.warning(
                f"Exception creating fallback coords! Discarding any inferred anchor keypoints - {repr(ex)}"
            )

        return lons, lats

    def _get_lonlat_per_pixel(
        self,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
        geofence_lonlat_center: Tuple[float, float],
        map_scale: Optional[MapScale],
    ) -> Tuple[float, float]:
        """
        Estimate the degrees-per-pixel resolution
        """

        lon_pt = (
            next(iter(lons.values())).get_parsed_degree()
            if len(lons) > 0
            else geofence_lonlat_center[0]
        )
        lat_pt = (
            next(iter(lats.values())).get_parsed_degree()
            if len(lats) > 0
            else geofence_lonlat_center[1]
        )
        lonlat_per_km = ScaleAnalyzer.calc_deg_per_km((lon_pt, lat_pt))

        if get_distinct_degrees(lons) > 1:
            # estimate from extracted lon keypoints
            deg_xy_pts = [
                (c.get_parsed_degree(), c.get_pixel_alignment()) for c in lons.values()
            ]
            # get lon_pts with min/max x-pixel values
            max_pt = max(deg_xy_pts, key=lambda p: p[1][0])
            min_pt = min(deg_xy_pts, key=lambda p: p[1][0])
            pxl_range = max_pt[1][0] - min_pt[1][0]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                # estimate the km-per-pixel
                km_per_pxl = abs((deg_range / lonlat_per_km[0]) / pxl_range)
                return (
                    lonlat_per_km[0] * km_per_pxl,
                    lonlat_per_km[1] * km_per_pxl,
                )

        if get_distinct_degrees(lats) > 1:
            # estimate from extracted lat keypoints
            deg_xy_pts = [
                (c.get_parsed_degree(), c.get_pixel_alignment()) for c in lats.values()
            ]
            # get lat_pts with min/max y-pixel values
            max_pt = max(deg_xy_pts, key=lambda p: p[1][1])
            min_pt = min(deg_xy_pts, key=lambda p: p[1][1])
            pxl_range = max_pt[1][1] - min_pt[1][1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                # estimate the km-per-pixel
                km_per_pxl = abs((deg_range / lonlat_per_km[1]) / pxl_range)
                return (
                    lonlat_per_km[0] * km_per_pxl,
                    lonlat_per_km[1] * km_per_pxl,
                )

        # estimate from ScaleAnalyzer info, or use fallback DPI and scale values
        dpi = 300
        scale_pixels = 100000
        lonlat_per_pixel = (0.0, 0.0)
        if map_scale:
            dpi = map_scale.dpi
            if map_scale.scale_pixels != 0:
                scale_pixels = map_scale.scale_pixels
                lonlat_per_pixel = map_scale.lonlat_per_pixel
        if lonlat_per_pixel[0] == 0:
            km_per_pixel = scale_pixels * KM_PER_INCH / dpi
            lonlat_per_pixel = (
                lonlat_per_km[0] * km_per_pixel,
                lonlat_per_km[1] * km_per_pixel,
            )
        return lonlat_per_pixel

    def _create_anchor_keypoints(
        self,
        coords: Dict[Tuple[float, float], Coordinate],
        geofence_center_deg: float,
        deg_per_pixel: float,
        map_poly: Polygon,
        is_lat: bool,
        force_deg_abs: bool = True,
    ) -> Dict[Tuple[float, float], Coordinate]:
        """
        Create fallback anchor keypoints, as needed
        (note, pixel coords are jittered by 1 pixel to help with polynomial geo-transform conditioning)
        """
        if get_distinct_degrees(coords) > 1:
            # no anchor keypoints needed
            return coords

        p_idx = 1 if is_lat else 0
        deg_anchor_pt = (
            abs(geofence_center_deg) if force_deg_abs else geofence_center_deg
        )
        xy_anchor_pt = (map_poly.centroid.x, map_poly.centroid.y)
        if len(coords) > 0:
            c = next(iter(coords.values()))
            deg_anchor_pt = c.get_parsed_degree()
            xy_anchor_pt = c.get_pixel_alignment()

        # --- top-left of map ROI
        xy = (map_poly.bounds[0], map_poly.bounds[1])
        deg = deg_per_pixel * (xy[p_idx] - xy_anchor_pt[p_idx]) + deg_anchor_pt
        c = Coordinate(
            CoordType.DERIVED_KEYPOINT,
            f"fallback {deg}",
            deg,
            CoordSource.ANCHOR,
            is_lat,
            pixel_alignment=xy,
            confidence=0.0,
        )
        coords = add_coordinate_to_dict(c, coords)
        # --- top-right of map ROI (jitter y by 1 pixel)
        xy = (map_poly.bounds[2], map_poly.bounds[1] + 1)
        deg = deg_per_pixel * (xy[p_idx] - xy_anchor_pt[p_idx]) + deg_anchor_pt
        c = Coordinate(
            CoordType.DERIVED_KEYPOINT,
            f"fallback {deg}",
            deg,
            CoordSource.ANCHOR,
            is_lat,
            pixel_alignment=xy,
            confidence=0.0,
        )
        coords = add_coordinate_to_dict(c, coords)
        # --- bottem-right of map ROI
        xy = (map_poly.bounds[2] - 1, map_poly.bounds[3])
        deg = deg_per_pixel * (xy[p_idx] - xy_anchor_pt[p_idx]) + deg_anchor_pt
        c = Coordinate(
            CoordType.DERIVED_KEYPOINT,
            f"fallback {deg}",
            deg,
            CoordSource.ANCHOR,
            is_lat,
            pixel_alignment=xy,
            confidence=0.0,
        )
        coords = add_coordinate_to_dict(c, coords)
        # --- bottem-left of map ROI (jitter x by 1 pixel)
        xy = (map_poly.bounds[0] + 1, map_poly.bounds[3] - 1)
        deg = deg_per_pixel * (xy[p_idx] - xy_anchor_pt[p_idx]) + deg_anchor_pt
        c = Coordinate(
            CoordType.DERIVED_KEYPOINT,
            f"fallback {deg}",
            deg,
            CoordSource.ANCHOR,
            is_lat,
            pixel_alignment=xy,
            confidence=0.0,
        )
        coords = add_coordinate_to_dict(c, coords)

        return coords
