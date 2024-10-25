import logging
from copy import deepcopy
from shapely import box, Point, Polygon
import numpy as np

from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
    SOURCE_GEOCODE,
)
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedPlace,
    GeocodedCoordinate,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import Dict, List, Tuple

COORDINATE_CONFIDENCE_GEOCODE = 0.8
MIN_KEYPOINTS = 4
IQR_FACTOR = 0.75  # 1.5  # interquartile range factor (for outlier analysis)

logger = logging.getLogger(__name__)


class PointGeocoder(CoordinatesExtractor):
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

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        """
        Main point geocoding extraction function
        """

        # get any existing lat / lon extractions for this image
        lon_pts: Dict[Tuple[float, float], Coordinate] = input.input.get_data("lons")
        lat_pts: Dict[Tuple[float, float], Coordinate] = input.input.get_data("lats")
        num_keypoints = min(len(lon_pts), len(lat_pts))

        # get existing geocoded places
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geoplaces = []
        if geocoded:
            # get the geocoded places for desired place_types (eg point and population centres)
            geoplaces = [
                p for p in geocoded.places if p.place_type in self._place_types
            ]
        if not geoplaces:
            logger.info("No point-based geocoded places; skipping point geocoder")
            return lon_pts, lat_pts

        # get geofence
        geofence: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        # prune places not inside the geofence
        geoplaces = self._geofence_filtering(geofence, geoplaces)
        logger.info(
            f"Number of geoplaces found within target geofence: {len(geoplaces)}"
        )

        # get the coordinates for the geoplaces
        coord_tuples = self._convert_to_coordinates(geoplaces, self._use_abs)

        # do quantile filtering
        points_thres = self.points_thres
        if (
            geofence.geofence.region_type == GeoFenceType.DEFAULT
            or geofence.geofence.region_type == GeoFenceType.COUNTRY
        ):
            # use a lower points threshold if the geofence is very coarse
            points_thres = 5
        coord_tuples = self._quantile_filtering(coord_tuples, points_thres)
        # do confidence filtering
        coord_tuples = self._confidence_filtering(coord_tuples, num_keypoints)

        # append to main lon/lat results
        logger.info(
            f"Number of final lat/lon coords found from point geocoding: {len(coord_tuples)}"
        )
        for lon_coord, lat_coord in coord_tuples:
            lon_pts[lon_coord.to_deg_result()[0]] = lon_coord
            lat_pts[lat_coord.to_deg_result()[0]] = lat_coord

        return lon_pts, lat_pts

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
                "point derived lon",
                p.place_name,
                lon_val,
                SOURCE_GEOCODE,
                False,
                pixel_alignment=pixel_alignment,
                confidence=confidence,
                derivation="geocoded",
            )
            lat_pt = Coordinate(
                "point derived lat",
                p.place_name,
                lat_val,
                SOURCE_GEOCODE,
                True,
                pixel_alignment=pixel_alignment,
                confidence=confidence,
                derivation="geocoded",
            )

            pixels.add(pixel_alignment)
            coord_tuples.append((lon_pt, lat_pt))

        return coord_tuples

    def _quantile_filtering(
        self, coord_tuples: List[Tuple[Coordinate, Coordinate]], points_thres: int
    ) -> List[Tuple[Coordinate, Coordinate]]:
        """
        quantile filtering of lon and lat pairs to remove outliers
        (similar to box / whisker quantile analysis)
        """
        if len(coord_tuples) <= points_thres:
            return coord_tuples

        lon_arr = np.array([x[0].get_parsed_degree() for x in coord_tuples])
        lon_25 = np.percentile(lon_arr, 25)  # q1
        lon_75 = np.percentile(lon_arr, 75)  # q3
        lon_iqr_thres = (lon_75 - lon_25) * IQR_FACTOR

        lat_arr = np.array([x[1].get_parsed_degree() for x in coord_tuples])
        lat_25 = np.percentile(lat_arr, 25)  # q1
        lat_75 = np.percentile(lat_arr, 75)  # q3
        lat_iqr_thres = (lat_75 - lat_25) * IQR_FACTOR

        coords_filtered = []
        for c_lon, c_lat in coord_tuples:
            lon = c_lon.get_parsed_degree()
            lat = c_lat.get_parsed_degree()
            if (lon_25 - lon_iqr_thres <= lon <= lon_75 + lon_iqr_thres) and (
                lat_25 - lat_iqr_thres <= lat <= lat_75 + lat_iqr_thres
            ):
                coords_filtered.append((c_lon, c_lat))
        return coords_filtered

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
