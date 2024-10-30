import logging

from copy import deepcopy
from geopy.distance import distance as geo_distance

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.geo_referencing.entities import (
    DocGeoFence,
    GeoFence,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
)
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeoPlaceType,
    GeoFeatureType,
    GEOCODED_PLACES_OUTPUT_KEY,
)

from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

CLUE_POINT_GEOFENCE_RANGE = 500


class GeoFencer(Task):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:

        clue_point = input.get_request_info("clue_point")
        if clue_point is not None:
            geofence = self._get_clue_point_geofence(
                input.raster_id, clue_point, CLUE_POINT_GEOFENCE_RANGE
            )
            return self._create_result(input, geofence)

        geocoded: DocGeocodedPlaces = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )

        geofence = self._get_geofence(input, geocoded)

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
                region_type=GeoFenceType.CLUE,
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
    ) -> DocGeoFence:
        """
        Main geofence extraction function
        Try to construct using counties, then states, then the whole country
        """
        # use default if nothing geocoded
        if geocoded is None or len(geocoded.places) == 0:
            return DocGeoFence(
                map_id=input.raster_id, geofence=self._create_default_geofence()
            )

        # --- 1. using counties
        geofence = self._create_geofence(geocoded, GeoFeatureType.COUNTY)

        # --- 2. using states
        if not geofence:
            geofence = self._create_geofence(geocoded, GeoFeatureType.STATE)

        # --- 3. using country
        if not geofence:
            geofence = self._create_geofence(geocoded, GeoFeatureType.COUNTRY)

        if geofence:
            return DocGeoFence(map_id=input.raster_id, geofence=geofence)
        else:
            return DocGeoFence(
                map_id=input.raster_id, geofence=self._create_default_geofence()
            )

    def _create_default_geofence(self) -> GeoFence:
        """
        Create geofence from the default ranges (as specified as pipeline input parameters),
        or alternatively, set to the geofence to the whole world
        """
        lon_minmax = input.get_request_info("lon_minmax", [-180, 180])
        lat_minmax = input.get_request_info("lat_minmax", [-90, 90])
        logger.info("Geofence created using defaults")
        return GeoFence(
            lat_minmax=deepcopy(lat_minmax),
            lon_minmax=deepcopy(lon_minmax),
            region_type=GeoFenceType.DEFAULT,
        )

    def _create_geofence(
        self, geocoded: DocGeocodedPlaces, geo_feature_type: GeoFeatureType
    ) -> Optional[GeoFence]:
        """
        Create a geofence based on a given geo-feature-type (county, state, or country)
        """
        lats = []
        lons = []
        places = []
        for p in geocoded.places:
            if (
                p.place_type == GeoPlaceType.BOUND
                and p.feature_type == geo_feature_type
            ):
                # extract all lat and lon
                lats = lats + [
                    p.results[0].coordinates[0].geo_y,
                    p.results[0].coordinates[2].geo_y,
                ]
                lons = lons + [
                    p.results[0].coordinates[0].geo_x,
                    p.results[0].coordinates[2].geo_x,
                ]
                places.append(p.place_name)
        if len(lats) != 0:

            fence_type = GeoFenceType.DEFAULT
            if geo_feature_type == GeoFeatureType.COUNTY:
                fence_type = GeoFenceType.COUNTY
            elif geo_feature_type == GeoFeatureType.STATE:
                fence_type = GeoFenceType.STATE
            elif geo_feature_type == GeoFeatureType.COUNTRY:
                fence_type = GeoFenceType.COUNTRY

            logger.info(f"Geofence created using these places: {places}")
            return GeoFence(
                lat_minmax=[min(lats), max(lats)],
                lon_minmax=[min(lons), max(lons)],
                region_type=fence_type,
            )
        else:
            return None
