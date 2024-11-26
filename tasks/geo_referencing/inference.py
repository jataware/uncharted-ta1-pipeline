import logging
from tasks.common.task import Task, TaskInput, TaskResult
from shapely import Polygon
from tasks.geo_referencing.entities import (
    DocGeoFence,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
    ROI_MAP_OUTPUT_KEY,
    MapROI,
    Coordinate,
    CoordSource,
    CoordType,
)
from tasks.geo_referencing.util import calc_lonlat_slope_signs
from tasks.common.task import Task, TaskInput, TaskResult

logger = logging.getLogger("inference_extractor")


class InferenceCoordinateExtractor(Task):
    """
    Use existing lat/lon extractions and map scale information to infer additional ground control point coords
    This can help "anchor" the polynomial transform to prevent erratic georef results
    """

    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """
        # TODO -- could try using the map scale info (km_per_pixel estimate)
        # to aid coord inference?

        # get the extracted coordinates
        lon_pts = input.get_data("lons", {})
        lat_pts = input.get_data("lats", {})

        if len(lon_pts) == 0 and len(lat_pts) == 0:
            # No coords available; skip inference
            return self._create_result(input)

        # get geofence result
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        lonlat_slope_signs = (
            -1,
            1,
        )  # (default deg->pixel slope directions are for hemispheres = West, North)
        if geofence and not geofence.geofence.region_type == GeoFenceType.DEFAULT:
            lonlat_slope_signs = calc_lonlat_slope_signs(
                geofence.geofence.lonlat_hemispheres
            )

        # get map-roi result
        map_roi = MapROI.model_validate(input.data[ROI_MAP_OUTPUT_KEY])
        if not map_roi:
            # No map ROI available; skip inference
            return self._create_result(input)
        map_poly = Polygon(map_roi.map_bounds)

        # check number of unique lat and lon values
        num_distinct_lats = len(set([x.get_parsed_degree() for x in lat_pts.values()]))
        num_distinct_lons = len(set([x.get_parsed_degree() for x in lon_pts.values()]))

        try:
            if num_distinct_lats >= 2 and num_distinct_lons == 1:
                # infer an additional lon pt (based on lat pxl resolution)
                logger.info("Inferring longitude coordinate from latitudes")
                # list of (deg, (x,y))
                deg_xy_pts = [
                    (c.get_parsed_degree(), c.get_pixel_alignment())
                    for c in lat_pts.values()
                ]
                # get lat_pts with min/max y-pixel values
                max_pt = max(deg_xy_pts, key=lambda p: p[1][1])
                min_pt = min(deg_xy_pts, key=lambda p: p[1][1])
                # estimate degree-per-pixel
                pxl_range = max_pt[1][1] - min_pt[1][1]
                deg_range = max_pt[0] - min_pt[0]
                if deg_range != 0 and pxl_range != 0:
                    deg_per_pxl = abs(deg_range / pxl_range)
                    existing_lon_pt = next(iter(lon_pts.values()))
                    map_pxl_mid = (map_poly.bounds[0] + map_poly.bounds[2]) / 2
                    # put inferred x pixel value in one of the map corners
                    new_x_pxl = (
                        map_poly.bounds[0]
                        if existing_lon_pt.get_pixel_alignment()[0] > map_pxl_mid
                        else map_poly.bounds[2]
                    )
                    # inferred y-pixel same as the existing lon pt, jittered by 1 pixel
                    new_y_pxl = max(existing_lon_pt.get_pixel_alignment()[1] - 1, 0)
                    new_lon = (
                        lonlat_slope_signs[0]
                        * deg_per_pxl
                        * (new_x_pxl - existing_lon_pt.get_pixel_alignment()[0])
                        + existing_lon_pt.get_parsed_degree()
                    )
                    # save inferred lon coordinate
                    lon_coord = Coordinate(
                        CoordType.DERIVED_KEYPOINT,
                        "",
                        new_lon,
                        CoordSource.INFERENCE,
                        False,
                        pixel_alignment=(
                            new_x_pxl,
                            new_y_pxl,
                        ),
                        confidence=0.5,
                    )
                    lon_pts[lon_coord.to_deg_result()[0]] = lon_coord

            elif num_distinct_lons >= 2 and num_distinct_lats == 1:
                # infer an additional lat pt (based on lon pxl resolution)
                logger.info("Inferring latitude coordinate from longitudes")
                # list of (deg, (x,y))
                deg_xy_pts = [
                    (c.get_parsed_degree(), c.get_pixel_alignment())
                    for c in lon_pts.values()
                ]
                # get lon_pts with min/max x-pixel values
                max_pt = max(deg_xy_pts, key=lambda p: p[1][0])
                min_pt = min(deg_xy_pts, key=lambda p: p[1][0])
                # estimate degree-per-pixel
                pxl_range = max_pt[1][0] - min_pt[1][0]
                deg_range = max_pt[0] - min_pt[0]
                if deg_range != 0 and pxl_range != 0:
                    deg_per_pxl = abs(deg_range / pxl_range)
                    existing_lat_pt = next(iter(lat_pts.values()))
                    map_pxl_mid = (map_poly.bounds[1] + map_poly.bounds[3]) / 2
                    # put inferred y pixel value in one of the map corners
                    new_y_pxl = (
                        map_poly.bounds[1]
                        if existing_lat_pt.get_pixel_alignment()[1] > map_pxl_mid
                        else map_poly.bounds[3]
                    )
                    # inferred x-pixel same as the existing lat pt, jittered by 1 pixel
                    new_x_pxl = max(existing_lat_pt.get_pixel_alignment()[0] - 1, 0)
                    new_lat = (
                        lonlat_slope_signs[1]
                        * deg_per_pxl
                        * (new_y_pxl - existing_lat_pt.get_pixel_alignment()[1])
                        + existing_lat_pt.get_parsed_degree()
                    )
                    # save inferred lat coordinate
                    lat_coord = Coordinate(
                        CoordType.DERIVED_KEYPOINT,
                        "",
                        new_lat,
                        CoordSource.INFERENCE,
                        True,
                        pixel_alignment=(
                            new_x_pxl,
                            new_y_pxl,
                        ),
                        confidence=0.5,
                    )
                    lat_pts[lat_coord.to_deg_result()[0]] = lat_coord

        except Exception as e:
            logger.warning(
                f"Exception in coord inference; discarding any inferred coords - {repr(e)}"
            )

        # update the coordinates lists
        result = self._create_result(input)
        result.output["lons"] = lon_pts
        result.output["lats"] = lat_pts
        return result
