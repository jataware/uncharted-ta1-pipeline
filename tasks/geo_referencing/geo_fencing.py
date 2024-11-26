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

# minmimum field-of-view for geo-fence
FOV_RANGE_KM = 500


class GeoFencer(Task):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:

        clue_point = input.get_request_info("clue_point")
        if clue_point is not None:
            geofence = self._get_clue_point_geofence(
                input.raster_id, clue_point, FOV_RANGE_KM
            )
            return self._create_result(input, geofence)

        geocoded: DocGeocodedPlaces = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )

        geofence = self._get_geofence(input, geocoded)

        # update the coordinates list
        return self._create_result(input, geofence)

    def _get_clue_point_geofence(
        self, raster_id: str, clue_point: Tuple[float, float], fov_range_km: int
    ) -> DocGeoFence:

        fov_degrange_lon, fov_degrange_lat = self._calc_fov_degree_ranges(
            clue_point, fov_range_km
        )
        lon_minmax = [
            clue_point[0] - fov_degrange_lon,
            clue_point[0] + fov_degrange_lon,
        ]
        lat_minmax = [
            clue_point[1] - fov_degrange_lat,
            clue_point[1] + fov_degrange_lat,
        ]
        lon_hemisphere = self._calc_hemisphere_multiplier(lon_minmax[0], lon_minmax[1])
        lat_hemisphere = self._calc_hemisphere_multiplier(lat_minmax[0], lat_minmax[1])

        return DocGeoFence(
            map_id=raster_id,
            geofence=GeoFence(
                lat_minmax=lat_minmax,
                lon_minmax=lon_minmax,
                region_type=GeoFenceType.CLUE,
                lonlat_hemispheres=(lon_hemisphere, lat_hemisphere),
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
                map_id=input.raster_id, geofence=self._create_default_geofence(input)
            )

        # --- 1. geofence using states
        geofence = self._create_geofence(geocoded, GeoFeatureType.STATE)
        if geofence:
            # --- 2. try to narrow the geofence using county names, if available
            geofence_county = self._create_geofence(geocoded, GeoFeatureType.COUNTY)
            if geofence_county:
                # double-check that the narrowed geofence isn't too small
                geofence = self._check_geofence_fov(
                    geofence_county, geofence, FOV_RANGE_KM
                )

        if not geofence:
            # --- 3. geofence using country
            geofence = self._create_geofence(geocoded, GeoFeatureType.COUNTRY)

        if geofence:
            return DocGeoFence(map_id=input.raster_id, geofence=geofence)
        else:
            return DocGeoFence(
                map_id=input.raster_id, geofence=self._create_default_geofence(input)
            )

    def _create_default_geofence(self, input: TaskInput) -> GeoFence:
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
            lonlat_hemispheres=(1, 1),
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

            lon_hemisphere = self._calc_hemisphere_multiplier(min(lons), min(lons))
            lat_hemisphere = self._calc_hemisphere_multiplier(min(lats), max(lats))

            logger.info(f"Geofence created using these places: {places}")
            return GeoFence(
                lat_minmax=[min(lats), max(lats)],
                lon_minmax=[min(lons), max(lons)],
                region_type=fence_type,
                lonlat_hemispheres=(lon_hemisphere, lat_hemisphere),
            )
        else:
            return None

    def _check_geofence_fov(
        self, geo_fence_narrow: GeoFence, geo_fence: GeoFence, fov_range_km: int
    ) -> GeoFence:
        """
        ensure the narrowed geo-fence field-of-view is not too small,
        and it is inside the wider geo-fence
        """
        if (
            geo_fence_narrow.lon_minmax[0] < geo_fence.lon_minmax[0]
            or geo_fence_narrow.lon_minmax[1] > geo_fence.lon_minmax[1]
            or geo_fence_narrow.lat_minmax[0] < geo_fence.lat_minmax[0]
            or geo_fence_narrow.lat_minmax[1] > geo_fence.lat_minmax[1]
        ):
            logger.info("Narrowed geofence is outside its parent; using wider geofence")
            return geo_fence

        # centre-point of geofence
        centre_lonlat = (
            (geo_fence_narrow.lon_minmax[0] + geo_fence_narrow.lon_minmax[1]) / 2,
            (geo_fence_narrow.lat_minmax[0] + geo_fence_narrow.lat_minmax[1]) / 2,
        )

        fov_degrange_lon, fov_degrange_lat = self._calc_fov_degree_ranges(
            centre_lonlat, fov_range_km
        )

        # ensure the narrowed geo-fence's FOV is not too small, but doesn't extend beyond the 'parent' geo_fence
        lat_minmax = geo_fence_narrow.lat_minmax
        lat_minmax[0] = max(
            min(lat_minmax[0], centre_lonlat[1] - fov_degrange_lat),
            geo_fence.lat_minmax[0],
        )
        lat_minmax[1] = min(
            max(lat_minmax[1], centre_lonlat[1] + fov_degrange_lat),
            geo_fence.lat_minmax[1],
        )
        geo_fence_narrow.lat_minmax = lat_minmax

        lon_minmax = geo_fence_narrow.lon_minmax
        lon_minmax[0] = max(
            min(lon_minmax[0], centre_lonlat[0] - fov_degrange_lon),
            geo_fence.lon_minmax[0],
        )
        lon_minmax[1] = min(
            max(lon_minmax[1], centre_lonlat[0] + fov_degrange_lon),
            geo_fence.lon_minmax[1],
        )
        geo_fence_narrow.lon_minmax = lon_minmax

        return geo_fence_narrow

    def _calc_fov_degree_ranges(
        self, centre_lonlat: Tuple[float, float], fov_range_km: int
    ) -> Tuple[float, float]:
        """
        calculate the desired field-of-view lon and lat degree range for a given KM range
        """

        dist_km = fov_range_km / 2.0
        fov_pt_north = geo_distance(kilometers=dist_km).destination(
            (centre_lonlat[1], centre_lonlat[0]), bearing=0
        )
        fov_pt_east = geo_distance(kilometers=dist_km).destination(
            (centre_lonlat[1], centre_lonlat[0]), bearing=90
        )
        fov_degrange_lon = abs(fov_pt_east[1] - centre_lonlat[0])
        fov_degrange_lat = abs(fov_pt_north[0] - centre_lonlat[1])
        return (fov_degrange_lon, fov_degrange_lat)

    def _calc_hemisphere_multiplier(self, coord_mid: float, coord_max: float) -> int:
        """
        Returns +1 for Northern latitudes or Eastern longitudes,
        returns -1 for Southern latitudes or Western longitudes
        """
        # TODO -- could improve this to handle international dateline (cyclic math)
        # and/or crossing the equator
        mid_pt = (coord_mid + coord_max) / 2
        return 1 if mid_pt >= 0.0 else -1
