import logging
from copy import deepcopy
from shapely import box, Point, Polygon

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GeoFenceType,
    CoordType,
    CoordSource,
    GEOFENCE_OUTPUT_KEY,
)
from tasks.geo_referencing.util import get_num_keypoints, add_coordinate_to_dict
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedPlace,
    GeocodedCoordinate,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import List, Tuple

COORDINATE_CONFIDENCE_GEOCODE = 0.8
MIN_KEYPOINTS = 4

logger = logging.getLogger(__name__)


class PointGeocoder(Task):
    """
    Generate georeferencing lat and lon results from the extracted place and population centre geo coordinates
    """

    def __init__(
        self,
        task_id: str,
        place_types: List[str] = ["point", "population"],
        points_thres: int = 8,
        use_abs: bool = True,
    ):
        super().__init__(task_id)
        self._place_types = place_types  # which place types to use for geocoding
        self.points_thres = points_thres
        self._use_abs = use_abs  # use abs of lon/lat for geocoding results?

    def run(self, input: TaskInput) -> TaskResult:
        """
        run the task
        """
        # get existing geocoded places
        geocoded: DocGeocodedPlaces = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geoplaces = []
        if geocoded:
            # get the geocoded places for desired place_types (eg point and population centres)
            geoplaces = [
                p for p in geocoded.places if p.place_type in self._place_types
            ]
        if not geoplaces:
            logger.info(
                "No point-based geocoded places available; skipping PointGeocoder"
            )
            return self._create_result(input)

        # get coordinates so far
        lons = input.get_data("lons", {})
        lats = input.get_data("lats", {})

        num_keypoints = get_num_keypoints(lons, lats)
        if num_keypoints >= 2:
            logger.info(
                f"Min number of keypoints = {num_keypoints}; skipping PointGeocoder"
            )
            return self._create_result(input)

        # get geofence
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        # prune places not inside the geofence
        geoplaces = self._geofence_filtering(geofence, geoplaces)
        logger.info(
            f"Number of geoplaces found within target geofence: {len(geoplaces)}"
        )

        # rank any duplicate results
        geoplaces = self._rank_duplicates(geoplaces)

        # get the coordinates for the geoplaces
        coord_tuples = self._convert_to_coordinates(geoplaces, self._use_abs)

        # do confidence filtering
        coord_tuples = self._confidence_filtering(coord_tuples, num_keypoints)

        # append to main lon/lat results
        logger.info(
            f"Number of final lat/lon coords found from point geocoding: {len(coord_tuples)}"
        )
        for lon_coord, lat_coord in coord_tuples:
            lons = add_coordinate_to_dict(lon_coord, lons)
            lats = add_coordinate_to_dict(lat_coord, lats)

        # update the coordinates lists
        result = self._create_result(input)
        result.output["lons"] = lons
        result.output["lats"] = lats

        return result

    def _geofence_filtering(
        self, geofence: DocGeoFence, geoplaces: List[GeocodedPlace]
    ) -> List[GeocodedPlace]:
        """
        prune places not inside the geofence
        """
        if not geofence or geofence.geofence.region_type == GeoFenceType.DEFAULT:
            return geoplaces

        geoplaces_filtered = []
        lon_minmax = geofence.geofence.lon_minmax
        lat_minmax = geofence.geofence.lat_minmax
        geofence_poly = box(lon_minmax[0], lat_minmax[0], lon_minmax[1], lat_minmax[1])

        for p in geoplaces:
            pc = deepcopy(p)
            coords = []
            for c in pc.results:
                in_geofence, dist = self._in_geofence(
                    self._point_to_coordinate(c.coordinates), geofence_poly
                )
                if in_geofence:
                    coords.append((c, dist))
            # sort by ascending distance to geofence centroid
            coords.sort(key=lambda x: x[1])
            if len(coords) > 0:
                pc.results = [c[0] for c in coords]
                geoplaces_filtered.append(pc)

        return geoplaces_filtered

    def _in_geofence(
        self,
        lonlat: Tuple[float, float],
        geofence_poly: Polygon,
    ) -> Tuple[bool, float]:
        """
        Returns True if a lon-lat point is within a geofence
        """
        pt = Point(lonlat[0], lonlat[1])
        if pt.intersects(geofence_poly):
            # calc distance to the geofence centroid
            # (TODO: ideally, could use geodesic distances here)
            dist = pt.distance(geofence_poly.centroid)
            return (True, dist)

        return (False, -1.0)

    def _point_to_coordinate(
        self, point: List[GeocodedCoordinate]
    ) -> Tuple[float, float]:
        """
        point to lon/lat tuple
        """
        return (point[0].geo_x, point[0].geo_y)

    def _rank_duplicates(self, geoplaces: List[GeocodedPlace]) -> List[GeocodedPlace]:
        """
        rank any duplicate geocoded places based on centroid of all geoplaces
        """

        do_ranking = any(len(p.results) > 1 for p in geoplaces)
        if not do_ranking:
            # no geoplace duplicates found; skip ranking
            return geoplaces
        try:
            # calc the weighted centroid of all geoplace points
            lons_sum = 0.0
            lats_sum = 0.0
            weight_sum = 0.0
            for p in geoplaces:
                weight = 1.0 / float(len(p.results))
                for c in p.results:
                    lonlat = self._point_to_coordinate(c.coordinates)
                    lons_sum += weight * lonlat[0]
                    lats_sum += weight * lonlat[1]
                    weight_sum += weight
            centroid = (lons_sum / weight_sum, lats_sum / weight_sum)
            centroid = Point(centroid)

            # sort duplicate geo-places by ascending distance to the centroid
            geoplaces_ranked = []
            for p in geoplaces:
                if len(p.results) > 1:
                    coords = []
                    for c in p.results:
                        lonlat = self._point_to_coordinate(c.coordinates)
                        pt = Point(lonlat[0], lonlat[1])
                        dist = pt.distance(centroid)
                        coords.append((c, dist))
                    coords.sort(key=lambda x: x[1])
                    p.results = [c[0] for c in coords]
                geoplaces_ranked.append(p)
        except Exception as e:
            logger.warning(
                f"Exception ranking duplicate geoplace results, {repr(e)}; skipping ranking operation"
            )
            return geoplaces

        return geoplaces_ranked

    def _convert_to_coordinates(
        self, places: List[GeocodedPlace], use_abs: bool = True
    ) -> List[Tuple[Coordinate, Coordinate]]:
        """
        Convert geocoded places to lon/lat coordinates
        Returns a list of Coordinate Tuples (lon, lat)
        """

        coord_tuples = []
        pixels = set()
        for p in places:

            confidence = COORDINATE_CONFIDENCE_GEOCODE
            # if p.place_type != "population":
            #     # slightly higher confidence for population centre geo-places
            #     confidence *= 0.8
            if len(p.results) > 1:
                # multiple lon/lat results for this place name!
                # (this may correspond to a river or road, etc.)
                # just take the first one and reduce the confidence
                confidence *= 0.5

            geocoord = p.results[0].coordinates[0]
            pixel_alignment = (geocoord.pixel_x, geocoord.pixel_y)
            if pixel_alignment in pixels:
                # already have a geocoded result for this pixel location; skip
                continue

            lon_val = abs(geocoord.geo_x) if use_abs else geocoord.geo_x
            lat_val = abs(geocoord.geo_y) if use_abs else geocoord.geo_y

            lon_pt = Coordinate(
                CoordType.GEOCODED_POINT,
                p.place_name,
                lon_val,
                CoordSource.GEOCODER,
                False,
                pixel_alignment=pixel_alignment,
                confidence=confidence,
            )
            lat_pt = Coordinate(
                CoordType.GEOCODED_POINT,
                p.place_name,
                lat_val,
                CoordSource.GEOCODER,
                True,
                pixel_alignment=pixel_alignment,
                confidence=confidence,
            )

            pixels.add(pixel_alignment)
            coord_tuples.append((lon_pt, lat_pt))

        return coord_tuples

    def _confidence_filtering(
        self,
        coord_tuples: List[Tuple[Coordinate, Coordinate]],
        num_existing_keypoints: int,
    ) -> List[Tuple[Coordinate, Coordinate]]:
        """
        confidence based filtering of geo-coding results
        """
        # ideally, we want at least 4 lat/lon pairs
        min_keypoints_needed = max(MIN_KEYPOINTS - num_existing_keypoints, 0)
        if min_keypoints_needed == 0:
            return []
        if len(coord_tuples) <= min_keypoints_needed:
            # coord pruning not needed
            return coord_tuples
        # sort by descending confidence
        coord_tuples.sort(key=lambda x: x[0].get_confidence(), reverse=True)

        # do confidence filtering
        confidence_thres = coord_tuples[min_keypoints_needed - 1][0].get_confidence()
        coord_tuples = list(
            filter(lambda x: x[0].get_confidence() >= confidence_thres, coord_tuples)
        )

        return coord_tuples
