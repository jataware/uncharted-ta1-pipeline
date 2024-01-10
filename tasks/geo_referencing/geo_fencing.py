from copy import deepcopy

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import DocGeoFence, GeoFence, GEOFENCE_OUTPUT_KEY
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import Optional


class GeoFencer(Task):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:
        geocoded: DocGeocodedPlaces = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )

        geofence = self._get_geofence(input, geocoded)

        # TODO: NEED TO DETERMINE IF INPUT HAS BETTER GEOFENCE (MAYBE BY AREA) OR SKIP IF RELATIVELY SMALL GEOFENCE

        # update the coordinates list
        return self._create_result(input, geofence)

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
    ) -> DocGeoFence:
        # use default if nothing geocoded
        if geocoded is None or len(geocoded.places) == None:
            return self._create_default_geofence(input)

        # for now, geofence should be either the widest possible to accomodate all states or if none present the country
        geofence = self._get_state_geofence(input, geocoded)
        if geofence is not None:
            return geofence

        return self._get_country_geofence(input, geocoded)

    def _create_default_geofence(self, input: TaskInput) -> DocGeoFence:
        lon_minmax = input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.get_request_info("lat_minmax", [0, 180])
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
    ) -> DocGeoFence:
        # country geofence is the one item with no restriction
        for p in geocoded.places:
            if (
                p.place_location_restriction is None
                or p.place_location_restriction == ""
            ):
                return DocGeoFence(
                    map_id=geocoded.map_id,
                    geofence=GeoFence(
                        lat_minmax=[p.coordinates[0].geo_y, p.coordinates[2].geo_y],
                        lon_minmax=[p.coordinates[0].geo_x, p.coordinates[2].geo_x],
                        defaulted=False,
                    ),
                )

        return self._create_default_geofence(input)

    def _get_state_geofence(
        self, input: TaskInput, geocoded: DocGeocodedPlaces
    ) -> Optional[DocGeoFence]:
        # for now, assume states have a restriction to the country
        lats = []
        lons = []
        for p in geocoded.places:
            if (
                p.place_location_restriction is not None
                and p.place_location_restriction != ""
            ):
                # extract all lat and lon
                lats = lats + [p.coordinates[0].geo_y, p.coordinates[2].geo_y]
                lons = lons + [p.coordinates[0].geo_x, p.coordinates[2].geo_x]
        if len(lats) == 0:
            return None

        # geofence is the widest possible box
        return DocGeoFence(
            map_id=input.raster_id,
            geofence=GeoFence(
                lat_minmax=[min(lats), max(lats)],
                lon_minmax=[min(lons), max(lons)],
                defaulted=False,
            ),
        )
