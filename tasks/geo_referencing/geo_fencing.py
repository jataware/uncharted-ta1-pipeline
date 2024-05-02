import logging

from copy import deepcopy
from geopy.distance import distance as geo_distance

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import DocGeoFence, GeoFence, GEOFENCE_OUTPUT_KEY
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import List, Optional, Tuple

logger = logging.getLogger("geo_fencer")

CLUE_POINT_GEOFENCE_RANGE = 500


class GeoFencer(Task):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        logger.info("running geo fencing task")

        clue_point = input.get_request_info("clue_point")
        if clue_point is not None:
            geofence = self._get_clue_point_geofence(
                input.raster_id, clue_point, CLUE_POINT_GEOFENCE_RANGE
            )
            self._add_param(
                input,
                "geo-fence-clue",
                "geofence",
                {
                    "clue-point": clue_point,
                    "geofence-lon": geofence.geofence.lon_minmax,
                    "geofence-lat": geofence.geofence.lat_minmax,
                },
                "geofence derived from clue point",
            )
            return self._create_result(input, geofence)

        geocoded: DocGeocodedPlaces = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )

        geofence, places = self._get_geofence(input, geocoded)
        self._add_param(
            input,
            "geo-fence-derived",
            "geofence",
            {
                "places": places,
                "geofence-lon": geofence.geofence.lon_minmax,
                "geofence-lat": geofence.geofence.lat_minmax,
            },
            "geofence derived from geocoded places",
        )

        # TODO: NEED TO DETERMINE IF INPUT HAS BETTER GEOFENCE (MAYBE BY AREA) OR SKIP IF RELATIVELY SMALL GEOFENCE

        # update the coordinates list
        return self._create_result(input, geofence)

    def _get_clue_point_geofence(
        self, raster_id: str, clue_point: Tuple[float, float], fov_range_km
    ) -> DocGeoFence:
        dist_km = (
            fov_range_km / 2.0
        )  # distance from clue pt in all directions (N,E,S,W)
        fov_pt_north = geo_distance(kilometers=dist_km).destination(
            (clue_point[1], clue_point[0]), bearing=0
        )
        fov_pt_east = geo_distance(kilometers=dist_km).destination(
            (clue_point[1], clue_point[0]), bearing=90
        )
        fov_degrange_lon = abs(fov_pt_east[1] - clue_point[0])
        fov_degrange_lat = abs(fov_pt_north[0] - clue_point[1])
        lon_minmax = [
            clue_point[0] - fov_degrange_lon,
            clue_point[0] + fov_degrange_lon,
        ]
        lat_minmax = [
            clue_point[1] - fov_degrange_lat,
            clue_point[1] + fov_degrange_lat,
        ]

        return DocGeoFence(
            map_id=raster_id,
            geofence=GeoFence(
                lat_minmax=lat_minmax,
                lon_minmax=lon_minmax,
                defaulted=False,
            ),
        )

    def _create_result(
        self,
        input: TaskInput,
        geofence: DocGeoFence,
    ) -> TaskResult:
        result = super()._create_result(input)
        result.add_output(GEOFENCE_OUTPUT_KEY, geofence.model_dump())

        return result

    def _get_geofence(
        self, input: TaskInput, geocoded: DocGeocodedPlaces
    ) -> Tuple[DocGeoFence, List[str]]:
        # use default if nothing geocoded
        if geocoded is None or len(geocoded.places) == None:
            return self._create_default_geofence(input), []

        # for now, geofence should be either the widest possible to accomodate all states or if none present the country
        geofence, places = self._get_state_geofence(input, geocoded)
        if geofence is not None:
            return geofence, places

        return self._get_country_geofence(input, geocoded)

    def _create_default_geofence(self, input: TaskInput) -> DocGeoFence:
        lon_minmax = input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.get_request_info("lat_minmax", [0, 90])
        return DocGeoFence(
            map_id=input.raster_id,
            geofence=GeoFence(
                lat_minmax=deepcopy(lat_minmax),
                lon_minmax=deepcopy(lon_minmax),
                defaulted=True,
            ),
        )

    def _get_country_geofence(
        self, input: TaskInput, geocoded: DocGeocodedPlaces
    ) -> Tuple[DocGeoFence, List[str]]:
        # country geofence is the one item with no restriction
        for p in geocoded.places:
            if (
                p.place_location_restriction is None
                or p.place_location_restriction == ""
            ):
                return DocGeoFence(
                    map_id=geocoded.map_id,
                    geofence=GeoFence(
                        lat_minmax=[
                            p.coordinates[0][0].geo_y,
                            p.coordinates[0][2].geo_y,
                        ],
                        lon_minmax=[
                            p.coordinates[0][0].geo_x,
                            p.coordinates[0][2].geo_x,
                        ],
                        defaulted=False,
                    ),
                ), [p.place_name]

        return self._create_default_geofence(input), []

    def _get_state_geofence(
        self, input: TaskInput, geocoded: DocGeocodedPlaces
    ) -> Tuple[Optional[DocGeoFence], List[str]]:
        # for now, assume states have a restriction to the country
        lats = []
        lons = []
        places = []
        for p in geocoded.places:
            if (
                p.place_location_restriction is not None
                and p.place_location_restriction != ""
            ) and p.place_type == "bound":
                # extract all lat and lon
                lats = lats + [p.coordinates[0][0].geo_y, p.coordinates[0][2].geo_y]
                lons = lons + [p.coordinates[0][0].geo_x, p.coordinates[0][2].geo_x]
                places.append(p.place_name)
        if len(lats) == 0:
            return None, []

        # geofence is the widest possible box
        return (
            DocGeoFence(
                map_id=input.raster_id,
                geofence=GeoFence(
                    lat_minmax=[min(lats), max(lats)],
                    lon_minmax=[min(lons), max(lons)],
                    defaulted=False,
                ),
            ),
            places,
        )
