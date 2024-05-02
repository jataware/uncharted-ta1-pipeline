import logging
import uuid

import numpy as np

from copy import deepcopy

from sklearn.cluster import DBSCAN

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GeoFence,
    GEOFENCE_OUTPUT_KEY,
)
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedPlace,
    GeocodedCoordinate,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import Dict, List, Optional, Tuple

COORDINATE_CONFIDENCE_GEOCODE = 0.8

logger = logging.getLogger("geocode")


class Geocoder(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        geocoded: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        lat_minmax, lon_minmax, defaulted = self._get_input_geofence(input)
        places = [p for p in geocoded.places if p.place_type == "point"]

        # filter places to only consider those within the geofence
        # TODO: may need to deep copy the object to not overwrite coordinates
        logger.info(
            f"extracting coordinates via geocoding with {len(places)} locations"
        )
        places_filtered = []
        if defaulted:
            places_filtered = places
        else:
            for p in places:
                pc = deepcopy(p)
                coords = []
                for c in pc.coordinates:
                    if self._in_geofence(
                        self._point_to_coordinate(c), lon_minmax, lat_minmax
                    ):
                        coords.append(c)
                if len(coords) > 0:
                    pc.coordinates = coords
                    places_filtered.append(pc)
                else:
                    logger.info(
                        f"removing {pc.place_name} from location set since no coordinates fall within the geofence"
                    )
        logger.info(
            f"extracting coordinates via geocoding with {len(places_filtered)} locations remaining after filtering"
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
            self._add_param(
                input.input,
                str(uuid.uuid4()),
                f"coordinate-{c.get_type()}-geocoded",
                {
                    "text": c.get_text(),
                    "parsed": c.get_parsed_degree(),
                    "type": "latitude" if c.is_lat() else "longitude",
                    "pixel_alignment": c.get_pixel_alignment(),
                    "confidence": c.get_confidence(),
                },
                "geocoded coordinate",
            )

        return lon_pts, lat_pts

    def _get_coordinates(self, places: List[GeocodedPlace]) -> List[Coordinate]:
        # cluster points using the geographic coordinates
        if len(places) == 0:
            return []
        coords = []
        for p in places:
            coords = coords + [
                ((c[0].geo_x, c[0].geo_y), (p, c[0].pixel_x, c[0].pixel_y))
                for c in p.coordinates
            ]
        data = np.array([c[0] for c in coords])  # .reshape(-1, 1)

        db = DBSCAN(eps=0.3, min_samples=3).fit(data)
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
        for c in max_cluster:
            coordinates.append(
                Coordinate(
                    "point derived lat",
                    c[1][0].place_name,
                    abs(c[0][1]),
                    True,
                    pixel_alignment=(c[1][1], c[1][2]),
                    confidence=COORDINATE_CONFIDENCE_GEOCODE,
                    derivation="geocoded",
                )
            )
            coordinates.append(
                Coordinate(
                    "point derived lon",
                    c[1][0].place_name,
                    abs(c[0][0]),
                    False,
                    pixel_alignment=(c[1][1], c[1][2]),
                    confidence=COORDINATE_CONFIDENCE_GEOCODE,
                    derivation="geocoded",
                )
            )
        return coordinates

    def _in_geofence(
        self,
        coordinate: Tuple[float, float],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> bool:
        # check x falls within geofence
        if not lon_minmax[0] <= abs(coordinate[0]) <= lon_minmax[1]:
            return False

        # check y falls within geofence
        if not lat_minmax[0] <= abs(coordinate[1]) <= lat_minmax[1]:
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
