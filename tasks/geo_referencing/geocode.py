import logging
import uuid

import numpy as np

from statistics import median

from copy import deepcopy

from sklearn.cluster import DBSCAN

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
    GeoPlaceType,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import Dict, List, Tuple, Callable

COORDINATE_CONFIDENCE_GEOCODE = 0.8

logger = logging.getLogger("geocode")


class Geocoder(CoordinatesExtractor):
    def __init__(self, task_id: str, place_types: List[GeoPlaceType]):
        super().__init__(task_id)
        self._place_types = place_types

    def _should_run(self, input: CoordinateInput) -> bool:
        return super()._should_run(input)

    def _filter_coordinates(
        self, geofence_raw: DocGeoFence, geocoded: DocGeocodedPlaces
    ) -> List[GeocodedPlace]:
        places = [p for p in geocoded.places if p.place_type in self._place_types]
        places_filtered = []
        if geofence_raw.geofence.region_type == GeoFenceType.DEFAULT:
            places_filtered = places
        else:
            lon_minmax = geofence_raw.geofence.lon_minmax
            lat_minmax = geofence_raw.geofence.lat_minmax
            for p in places:
                pc = deepcopy(p)
                coords = []
                for c in pc.results:
                    if self._in_geofence(
                        self._point_to_coordinate(c.coordinates), lon_minmax, lat_minmax
                    ):
                        coords.append(c)
                if len(coords) > 0:
                    pc.results = coords
                    places_filtered.append(pc)
                else:
                    logger.debug(
                        f"removing {pc.place_name} from location set since no coordinates fall within the geofence"
                    )
        return places_filtered

    def _get_coordinates(self, places: List[GeocodedPlace]) -> List[Coordinate]:
        # cluster points using the geographic coordinates
        if len(places) == 0:
            return []
        coords = []
        for p in places:
            coords = coords + [
                (
                    (c.coordinates[0].geo_x, c.coordinates[0].geo_y),
                    (p, c.coordinates[0].pixel_x, c.coordinates[0].pixel_y),
                )
                for c in p.results
            ]
        data = np.array([c[0] for c in coords])  # .reshape(-1, 1)

        db = DBSCAN(eps=1, min_samples=3).fit(data)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # find the biggest cluster (if any)
        clusters = []
        max_cluster = []
        for i, l in enumerate(labels):
            if l == -1:
                continue
            while len(clusters) <= l:
                clusters.append([])
            clusters[l].append(coords[i])
            if len(clusters[l]) > len(max_cluster):
                max_cluster = clusters[l]

        # create the coordinates for the clustered points (one for lon, one for lat)
        coordinates = []
        pixels = {}
        for c in max_cluster:
            pixel_alignment = (c[1][1], c[1][2])
            if pixel_alignment not in pixels:
                coordinates.append(
                    Coordinate(
                        "point derived lat",
                        c[1][0].place_name,
                        c[0][1],
                        SOURCE_GEOCODE,
                        True,
                        pixel_alignment=pixel_alignment,
                        confidence=COORDINATE_CONFIDENCE_GEOCODE,
                        derivation="geocoded",
                    )
                )
                coordinates.append(
                    Coordinate(
                        "point derived lon",
                        c[1][0].place_name,
                        c[0][0],
                        SOURCE_GEOCODE,
                        False,
                        pixel_alignment=pixel_alignment,
                        confidence=COORDINATE_CONFIDENCE_GEOCODE,
                        derivation="geocoded",
                    )
                )
                pixels[pixel_alignment] = 1
        return coordinates

    def _in_geofence(
        self,
        coordinate: Tuple[float, float],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> bool:
        # check x falls within geofence
        lons = [abs(x) for x in lon_minmax]
        lons = [min(lons), max(lons)]
        if not lons[0] <= abs(coordinate[0]) <= lons[1]:
            return False

        # check y falls within geofence
        lats = [abs(x) for x in lat_minmax]
        lats = [min(lats), max(lats)]
        if not lats[0] <= abs(coordinate[1]) <= lats[1]:
            return False
        return True

    def _point_to_coordinate(
        self, point: List[GeocodedCoordinate]
    ) -> Tuple[float, float]:
        return (point[0].geo_x, point[0].geo_y)

    def _get_point_geo(
        self, coordinate: List[GeocodedCoordinate]
    ) -> Tuple[float, float]:
        # assume a simple bounding box
        return (coordinate[0].geo_x + coordinate[2].geo_x) / 2, (
            coordinate[0].geo_y + coordinate[2].geo_y
        ) / 2


class PointGeocoder(Geocoder):
    def __init__(self, task_id: str, place_types: List[GeoPlaceType], run_limit: int):
        super().__init__(task_id, place_types)
        self._run_limit = run_limit

    def _should_run(self, input: CoordinateInput) -> bool:
        parent_should = super()._should_run(input)

        # dont run if the base condition is not met (sufficient coordinates already parsed)
        if not parent_should:
            return False

        # check how many geocoded points there are
        #   only run if a sufficiently large amount of them exist
        #   only run if a sufficiently large area of the map is covered
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geofence_raw: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )

        # filter places to only consider those within the geofence
        places_filtered = self._filter_coordinates(geofence_raw, geocoded)

        # TODO: CHECK FOR SPREAD ACROSS MAP AREA
        logger.debug(
            f"point geocoder to run if {len(places_filtered)} < {self._run_limit}"
        )
        return len(places_filtered) < self._run_limit

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geofence_raw: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )

        # filter places to only consider those within the geofence
        # TODO: may need to deep copy the object to not overwrite coordinates
        places_filtered = self._filter_coordinates(geofence_raw, geocoded)
        logger.info(
            f"extracting coordinates via point geocoding with {len(places_filtered)} locations remaining after filtering"
        )

        # get the coordinates for the points that fall within range
        coordinates = self._get_coordinates(places_filtered)

        # create the required coordinate structures
        lon_pts: Dict[Tuple[float, float], Coordinate] = input.input.get_data("lons")
        lat_pts: Dict[Tuple[float, float], Coordinate] = input.input.get_data("lats")
        for c in coordinates:
            d = lon_pts
            if c.is_lat():
                d = lat_pts
            d[c.to_deg_result()[0]] = c

        return lon_pts, lat_pts


class BoxGeocoder(Geocoder):
    def __init__(
        self, task_id: str, place_types: List[GeoPlaceType], run_limit: int = 10
    ):
        super().__init__(task_id, place_types)
        self._run_limit = run_limit

    def _should_run(self, input: CoordinateInput) -> bool:
        parent_should = super()._should_run(input)

        # dont run if the base condition is not met (sufficient coordinates already parsed)
        if not parent_should:
            return False

        # check how many geocoded points there are
        #   only run if a sufficiently large amount of them exist
        #   only run if a sufficiently large area of the map is covered
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geofence_raw: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )

        # filter places to only consider those within the geofence
        places_filtered = self._filter_coordinates(geofence_raw, geocoded)

        # TODO: CHECK FOR SPREAD ACROSS MAP AREA
        logger.debug(
            f"box geocoder to run if {len(places_filtered)} >= {self._run_limit}"
        )
        return len(places_filtered) >= self._run_limit

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        geofence_raw: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )
        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")

        # filter places to only consider those within the geofence
        places_filtered = self._filter_coordinates(geofence_raw, geocoded)
        logger.info(
            f"extracting coordinates via box geocoding with {len(places_filtered)} locations remaining after filtering"
        )

        # cluster to figure out which geocodings to consider
        coordinates = self._get_coordinates(places_filtered)
        logger.debug(f"got {len(coordinates)} geocoded coordinates for box geocoding")

        # keep the middle 80% of each direction roughly
        # a point dropped for one direction cannot be used for the other direction
        # TODO: SHOULD PROBABLY BE MORE BOX AND WHISKER STYLE OUTLIER FILTERING
        logger.debug(
            f"removing outlier coordinates via iqr from each end and direction"
        )
        coordinates_lons = self._remove_outliers(
            list(filter(lambda x: not x.is_lat(), coordinates)),
            lambda x: x.get_parsed_degree(),
        )
        coordinates_lats = self._remove_outliers(
            list(filter(lambda x: x.is_lat(), coordinates)),
            lambda x: x.get_parsed_degree(),
        )
        logger.info(
            f"after coordinate removal {len(coordinates_lons)} lons and {len(coordinates_lats)} lats remain"
        )

        # keep those that are in both lists only
        coordinates_count = {}
        for c in coordinates_lats:
            coordinates_count[c.get_pixel_alignment()] = 1
        count = 0
        for c in coordinates_lons:
            pixels = c.get_pixel_alignment()
            if pixels not in coordinates_count:
                coordinates_count[pixels] = 0
            coordinates_count[pixels] = coordinates_count[pixels] + 1
            count = count + 1

        coordinates_lons: List[Coordinate] = []
        coordinates_lats: List[Coordinate] = []
        for c in coordinates:
            pixels = c.get_pixel_alignment()
            if pixels in coordinates_count and coordinates_count[pixels] == 2:
                if c.is_lat():
                    coordinates_lats.append(c)
                else:
                    coordinates_lons.append(c)
        logger.debug(
            f"after harmonizing removals {len(coordinates_lons)} lons and {len(coordinates_lats)} lats remain"
        )

        # determine the pixel range and rough latitude / longitude range (assume x -> lon, y -> lat)
        min_lon, max_lon = self._get_min_max(coordinates_lons)
        min_lat, max_lat = self._get_min_max(coordinates_lats)
        logger.debug("obtained the coordinates covering the target range")
        coordinates_all = coordinates_lons + coordinates_lats
        vals_x = [c.get_pixel_alignment()[0] for c in coordinates_all]
        vals_y = [c.get_pixel_alignment()[1] for c in coordinates_all]
        min_x, max_x = min(vals_x), max(vals_x)
        min_y, max_y = min(vals_y), max(vals_y)
        logger.debug(
            f"creating coordinates between pixels x ({min_x}, {max_x}) and y ({min_y}, {max_y}) using lons {min_lon.get_parsed_degree()}, {max_lon.get_parsed_degree()} and lats {min_lat.get_parsed_degree()}, {max_lat.get_parsed_degree()}"
        )

        # create new points at the extremes of both ranges (ex: min x -> min lon, max x -> max lon)
        coords = self._create_coordinates(
            (min_x, max_x), (min_y, max_y), (min_lon, max_lon), (min_lat, max_lat)
        )
        for c in coords:
            if c.is_lat():
                lat_pts[c.to_deg_result()[0]] = c
            else:
                lon_pts[c.to_deg_result()[0]] = c

        return lon_pts, lat_pts

    def _remove_outliers(
        self, coordinates: List[Coordinate], mapper: Callable
    ) -> List[Coordinate]:
        # map and sort the coordinates
        values = sorted([(mapper(c), c) for c in coordinates], key=lambda x: x[0])

        # find the iqr
        lh_index = int(len(values) / 2)
        if len(values) % 2 == 0:
            lh_index = lh_index + 1
        q1 = median([v[0] for v in values[:lh_index]])
        q3 = median([v[0] for v in values[lh_index:]])
        iqr = q3 - q1

        # use usual iqr * 1.5 as basis for filtering
        remaining_values = [
            cf[1] for cf in values if q1 - (iqr * 1.5) <= cf[0] <= q3 + (iqr * 1.5)
        ]
        return remaining_values

    def _get_min_max(
        self, coordinates: List[Coordinate]
    ) -> Tuple[Coordinate, Coordinate]:
        # find the min & max coordinate
        degrees = [c.get_parsed_degree() for c in coordinates]
        deg_min, deg_max = min(degrees), max(degrees)
        coord_min, coord_max = None, None
        for c in coordinates:
            if c.get_parsed_degree() == deg_min:
                coord_min = c
            if c.get_parsed_degree() == deg_max:
                coord_max = c
        assert coord_min is not None
        assert coord_max is not None
        return coord_min, coord_max

    def _create_coordinates(
        self,
        minmax_x: Tuple[float, float],
        minmax_y: Tuple[float, float],
        minmax_lon: Tuple[Coordinate, Coordinate],
        minmax_lat: Tuple[Coordinate, Coordinate],
    ) -> List[Coordinate]:
        # for lon, min and max x always map together
        pixels_x = minmax_x

        # for lat, the mapping is reversed (min y -> max lat, max y -> min lat)
        pixels_y = (minmax_y[1], minmax_y[0])

        # create the four new coordinates mapping pixels to degrees
        return [
            Coordinate(
                "box derived lon",
                minmax_lon[0].get_text(),
                minmax_lon[0].get_parsed_degree(),
                SOURCE_GEOCODE,
                False,
                pixel_alignment=(pixels_x[0], pixels_y[1]),
                confidence=COORDINATE_CONFIDENCE_GEOCODE,
                derivation="geocoded",
            ),
            Coordinate(
                "box derived lon",
                minmax_lon[1].get_text(),
                minmax_lon[1].get_parsed_degree(),
                SOURCE_GEOCODE,
                False,
                pixel_alignment=(pixels_x[1], pixels_y[0]),
                confidence=COORDINATE_CONFIDENCE_GEOCODE,
                derivation="geocoded",
            ),
            Coordinate(
                "box derived lat",
                minmax_lat[0].get_text(),
                minmax_lat[0].get_parsed_degree(),
                SOURCE_GEOCODE,
                True,
                pixel_alignment=(pixels_x[0], pixels_y[0]),
                confidence=COORDINATE_CONFIDENCE_GEOCODE,
                derivation="geocoded",
            ),
            Coordinate(
                "box derived lat",
                minmax_lat[1].get_text(),
                minmax_lat[1].get_parsed_degree(),
                SOURCE_GEOCODE,
                True,
                pixel_alignment=(pixels_x[1], pixels_y[1]),
                confidence=COORDINATE_CONFIDENCE_GEOCODE,
                derivation="geocoded",
            ),
        ]
