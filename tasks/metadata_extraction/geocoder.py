import logging
import random
from shapely.geometry import Polygon, Point

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.metadata_extraction.geocoding_service import GeocodingService
from tasks.metadata_extraction.entities import (
    DocGeocodedPlaces,
    GeocodedCoordinate,
    GeocodedPlace,
    GeocodedResult,
    MetadataExtraction,
    GEOCODED_PLACES_OUTPUT_KEY,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.entities import (
    DocGeoFence,
    MapROI,
    GeoFenceType,
    GEOFENCE_OUTPUT_KEY,
    ROI_MAP_OUTPUT_KEY,
)
from tasks.text_extraction.entities import TextExtraction

from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Geocoder(Task):
    _geocoding_service: GeocodingService

    def __init__(
        self,
        task_id: str,
        geocoding_service: GeocodingService,
        run_bounds: bool = True,
        run_points: bool = True,
        run_centres: bool = True,
        should_run: Optional[Callable] = None,
    ):
        super().__init__(task_id)
        self._geocoding_service = geocoding_service
        self._run_bounds = run_bounds
        self._run_points = run_points
        self._run_centres = run_centres
        self._should_run = should_run

    def run(self, input: TaskInput) -> TaskResult:
        metadata: MetadataExtraction = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        geocoded_output = input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        if geocoded_output is None:
            geocoded_output = DocGeocodedPlaces(map_id=input.raster_id, places=[])

        if self._should_run and not self._should_run(input):
            logging.info("Skipping geocoding task")
            return self._create_result(input, geocoded_output)

        logger.info(f"running geocoding task with id {self._task_id}")
        # get any existing geocoded places
        geo_places = {str(place.place_name): place for place in geocoded_output.places}
        # get geofence if available and use to constrain the geocoder
        geofence = input.parse_data(GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate)
        # get the map region-of-interest, if available
        map_roi = input.parse_data(ROI_MAP_OUTPUT_KEY, MapROI.model_validate)

        # get new places to geocode
        to_geocode = self._get_places(metadata)
        # prune any point or population centre geoplaces that aren't in the map ROI
        to_geocode = self._roi_filtering(to_geocode, map_roi)
        # do geocoding...
        new_places = self._geocode_list(to_geocode, geofence)
        logger.info(f"geocoded {len(new_places)} places")

        # append to any existing geocoding results (use dict to prevent duplicates)
        for place in new_places:
            geo_places[place.place_name] = place
        geocoded_output.places = list(geo_places.values())

        # update the coordinates list
        return self._create_result(input, geocoded_output)

    def _roi_filtering(
        self, places: List[GeocodedPlace], map_roi: Optional[MapROI]
    ) -> List[GeocodedPlace]:
        """
        Prune point-based geoplaces if their pixel location isn't in the map ROI
        """
        if not map_roi:
            logger.warning(
                "No map ROI available, skipping ROI filteirng of geo-places."
            )
            return places
        if not self._run_centres and not self._run_points:
            # point-based geocoding disabled; filtering not needed
            return places

        try:
            map_poly = Polygon(map_roi.map_bounds)

            filtered_places = []
            for p in places:
                if self._run_points and p.place_type == "point":
                    # assuming singular point; check if pixel loc is inside map ROI
                    point = Point(
                        p.results[0].coordinates[0].pixel_x,
                        p.results[0].coordinates[0].pixel_y,
                    )
                    if point.intersects(map_poly):
                        filtered_places.append(p)
                elif self._run_centres and p.place_type == "population":
                    # assuming singular point; check if pixel loc is inside map ROI
                    point = Point(
                        p.results[0].coordinates[0].pixel_x,
                        p.results[0].coordinates[0].pixel_y,
                    )
                    if point.intersects(map_poly):
                        filtered_places.append(p)
                else:
                    filtered_places.append(p)

            return filtered_places

        except Exception as e:
            logger.error(f"Exception with _roi_filtering: {repr(e)}")
            return places

    def _geocode_list(
        self, to_geocode: List[GeocodedPlace], geofence: Optional[DocGeoFence]
    ) -> List[GeocodedPlace]:
        """
        Geocode a list of places
        """

        geobounds = None
        if (
            (self._run_points or self._run_centres)
            and geofence is not None
            and not geofence.geofence.region_type == GeoFenceType.DEFAULT
        ):
            # use the existing geofence bounds to constrain the geocoder
            lat_minmax = geofence.geofence.lat_minmax
            lon_minmax = geofence.geofence.lon_minmax
            geobounds = ((lat_minmax[0], lon_minmax[0]), (lat_minmax[1], lon_minmax[1]))

        # perform geocoding based on place type
        new_places = []
        if self._run_bounds:
            new_places = new_places + self._geocode_bounds(to_geocode)
        if self._run_points:
            new_places = new_places + self._geocode_points(to_geocode, geobounds)
        if self._run_centres:
            new_places = new_places + self._geocode_centres(to_geocode, geobounds)

        return new_places

    def _create_result(
        self,
        input: TaskInput,
        geocoded: DocGeocodedPlaces,
    ) -> TaskResult:
        result = super()._create_result(input)
        result.add_output(GEOCODED_PLACES_OUTPUT_KEY, geocoded.model_dump())

        return result

    def _get_places(self, metadata: MetadataExtraction) -> List[GeocodedPlace]:
        """
        Extract place-names to geocode from the metadata task result
        """
        if metadata is None:
            return []

        places: List[GeocodedPlace] = []
        country = ""
        if metadata.country and not metadata.country == "NULL":
            country = metadata.country
        if self._run_bounds:
            # init 'bounds' type geo-places (countries, states)
            if country:
                places.append(
                    GeocodedPlace(
                        place_name=country,
                        place_location_restriction="",
                        place_type="bound",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[
                                    GeocodedCoordinate(
                                        geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                    )
                                ],
                            )
                        ],
                    )
                )

            for s in metadata.states:
                if s and not s == "NULL":
                    places.append(
                        GeocodedPlace(
                            place_name=s,
                            place_location_restriction=country,
                            place_type="bound",
                            results=[
                                GeocodedResult(
                                    place_region="",
                                    coordinates=[
                                        GeocodedCoordinate(
                                            geo_x=0, geo_y=0, pixel_x=0, pixel_y=0
                                        )
                                    ],
                                )
                            ],
                        )
                    )

        if self._run_points:
            for p in metadata.places:
                if not isinstance(p, TextExtraction):
                    logger.error("place is not a text extraction")
                    continue
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="point",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[self._map_coordinates(p)],
                            )
                        ],
                    )
                )

        if self._run_centres:
            for p in metadata.population_centres:
                if not isinstance(p, TextExtraction):
                    logger.error("population centre is not a text extraction")
                    continue
                places.append(
                    GeocodedPlace(
                        place_name=p.text,
                        place_location_restriction=country,
                        place_type="population",
                        results=[
                            GeocodedResult(
                                place_region="",
                                coordinates=[self._map_coordinates(p)],
                            )
                        ],
                    )
                )

        return places

    def _reduce_to_point(self, place: TextExtraction) -> Tuple[float, float]:
        # for now reduce it to the central point
        polygon = Polygon([(b.x, b.y) for b in place.bounds])
        centroid = polygon.centroid
        return (centroid.x, centroid.y)

    def _geocode_bounds(self, places: List[GeocodedPlace]) -> List[GeocodedPlace]:
        """
        geocode bound places (eg countries, states)
        """
        geocoded_places = []
        bound_places_only = list(filter(lambda x: x.place_type == "bound", places))
        for p in bound_places_only:
            g, s = self._geocoding_service.geocode(p)
            if s:
                geocoded_places.append(g)
        return geocoded_places

    def _geocode_centres(
        self,
        places: List[GeocodedPlace],
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> List[GeocodedPlace]:
        """
        geocode population centres
        """
        geocoded_places = []
        pop_centres_only = list(filter(lambda x: x.place_type == "population", places))
        for p in pop_centres_only:
            g, s = self._geocoding_service.geocode(p, geofence=geofence)
            if s:
                geocoded_places.append(g)
        return geocoded_places

    def _geocode_points(
        self,
        places: List[GeocodedPlace],
        geofence: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> List[GeocodedPlace]:
        """
        geocode point places
        """
        geocoded_places = []
        points_only = list(filter(lambda x: x.place_type == "point", places))
        limit = min(200, len(points_only))
        places_to_geocode = random.sample(points_only, limit)
        for p in places_to_geocode:
            g, s = self._geocoding_service.geocode(p, geofence=geofence)
            if s:
                geocoded_places.append(g)
        return geocoded_places

    def _map_coordinates(self, extraction: TextExtraction) -> GeocodedCoordinate:
        # map from extraction coordinates to geocoded coordinates by using the centroid
        centroid = self._reduce_to_point(extraction)
        return GeocodedCoordinate(
            geo_x=0, geo_y=0, pixel_x=round(centroid[0]), pixel_y=round(centroid[1])
        )
