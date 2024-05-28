import copy
import logging
import re
import statistics
import utm
import uuid

from copy import deepcopy

from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.geo_referencing.entities import (
    Coordinate,
    DocGeoFence,
    GEOFENCE_OUTPUT_KEY,
    SOURCE_UTM,
)
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    DocGeocodedPlaces,
    GeocodedPlace,
    GEOCODED_PLACES_OUTPUT_KEY,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.util import (
    ocr_to_coordinates,
    get_bounds_bounding_box,
    is_in_range,
)

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("utm_extractor")

# UTM coordinates
# pre-compiled regex patterns
RE_NORTHEAST = re.compile(
    r"(?:^|\b)([1-9]?[ ,.]?[0-9]{3}[ ,.]?[0-9]00) ?m?(n|north|e|east)?($|\b)"
)

RE_NONNUMERIC = re.compile(r"[^0-9]")

RE_UTM_ZONE = re.compile(r"^utm zone ([1-9]{1,2})\s?([ns])($|\b)")

FOV_RANGE_METERS = 200000  # fallback search range (meters)

UTM_ZONE_DEFAULT = "default"
NORTHING_DEFAULT = "default"

EASTING_UPPER_LIMIT = 1000000  # easting values must be under a million


class UTMCoordinatesExtractor(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        geofence_raw: DocGeoFence = input.input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )

        ocr_blocks: DocTextExtraction = input.input.parse_data(
            TEXT_EXTRACTION_OUTPUT_KEY, DocTextExtraction.model_validate
        )
        metadata = input.input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        geocoded_places: DocGeocodedPlaces = input.input.parse_data(
            GEOCODED_PLACES_OUTPUT_KEY, DocGeocodedPlaces.model_validate
        )
        population_centres = [
            p for p in geocoded_places.places if p.place_type == "population"
        ]

        clue_point = input.input.get_request_info("clue_point")

        utm_zone = self._determine_utm_zone(
            metadata, population_centres, geofence_raw, clue_point
        )
        self._add_param(
            input.input,
            str(uuid.uuid4()),
            f"utm-zone",
            {
                "number": utm_zone[0],
                "northern": utm_zone[1],
                "source": utm_zone[2],
            },
            "extracted utm zone",
        )

        # lon_minmax = input.input.get_request_info("lon_minmax", [0, 180])
        # lat_minmax = input.input.get_request_info("lat_minmax", [-80, 84])
        # lon_minmax, lat_minmax, _ = self._get_input_geofence(input)
        lat_minmax = copy.deepcopy(geofence_raw.geofence.lat_minmax)
        lon_minmax = copy.deepcopy(geofence_raw.geofence.lon_minmax)

        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")

        # UTM is limited to the range of 80 south to 84 north
        if lat_minmax[0] < -80:
            lat_minmax[0] = -80
        if lat_minmax[1] > 84:
            lat_minmax[1] = 84
        logger.info(f"utm lon & lat limits: {lon_minmax}\t{lat_minmax}")
        logger.info(f"derived utm zone: {utm_zone}")

        lon_pts, lat_pts = self._extract_utm(
            input,
            utm_zone,
            ocr_blocks,
            (lon_pts, lat_pts),
            lon_minmax,
            lat_minmax,
        )

        return lon_pts, lat_pts

    def _is_northing_point(
        self,
        value: float,
        raw: str,
        dir_potential: str,
        parsed_direction: List[Tuple[float, bool]],
    ) -> Tuple[bool, str]:
        # returns true if the passed in value is a northing point, false if it is an easting point
        if dir_potential is None:
            dir_potential = ""
        dir_potential_lower = dir_potential.lower()
        if "e" in dir_potential_lower:
            return False, "parsed"
        if "n" in dir_potential_lower:
            return True, "parsed"

        # find the nearest neighbour as long as within order of magnitude if at least one in both directions
        north_east = map(lambda x: x[1], parsed_direction)
        if any([x for x in north_east]) and any([not x for x in north_east]):
            min_distance = abs(parsed_direction[0][0] - value)
            min_direction = parsed_direction[0][1]
            min_value = parsed_direction[0][0]
            for pd in parsed_direction:
                d = abs(pd[0] - value)
                if d < min_distance:
                    min_direction = pd[1]
                    min_distance = d
                    min_value = pd[0]
            if value / min_value > 0.1 and value / min_value < 10:
                return min_direction, "proximity"

        return value > 834000, NORTHING_DEFAULT

    def _determine_utm_zone(
        self,
        metadata: MetadataExtraction,
        geocoded_centres: List[GeocodedPlace],
        raw_geofence: DocGeoFence,
        clue_point: Optional[Tuple[float, float]],
    ) -> Tuple[int, bool, str]:
        # determine the UTM zone number and direction
        zone_number_determined = False
        zone_number = 1
        northern = False

        # check clue point first
        if clue_point is not None:
            # determine zone from lat & lon
            coord = utm.from_latlon(clue_point[1], clue_point[0])
            return coord[2], clue_point[1] > 0, "clue point"

        # check the utm zone for the number
        if (
            metadata.utm_zone is not None
            and len(metadata.utm_zone) > 0
            and metadata.utm_zone.isdigit()
        ):
            zone_number = int(metadata.utm_zone)
            zone_number_determined = True

        # extract the zone number from the coordinate systems
        for crs in metadata.coordinate_systems:
            # check for utm zone  DD[ns]
            lowered = crs.lower()
            for z in RE_UTM_ZONE.finditer(lowered):
                g = z.groups()
                if not zone_number_determined:
                    zone_number = int(g[0])
                northern = g[1] == "n"
                return zone_number, northern, "metadata crs"

        # if zone not specified in metadata, look to geocoded centres
        # TODO: MAYBE CHECK THAT THE GEOCODING IS WITHIN REASONABLE DISTANCE
        if len(geocoded_centres) > 0:
            if not zone_number_determined:
                zone_number = utm.latlon_to_zone_number(
                    geocoded_centres[0].coordinates[0][0].geo_y,
                    geocoded_centres[0].coordinates[0][0].geo_x,
                )
            northern = geocoded_centres[0].coordinates[0][0].geo_y > 0
            return zone_number, northern, "geocoding"

        # use geofence to determine the zone
        # set direction properly if min and max latitudes are in same hemisphere
        derived_direction = False
        unique_hemi = (
            raw_geofence.geofence.lat_minmax[0] * raw_geofence.geofence.lat_minmax[1]
        )
        if unique_hemi >= 0:
            hemi = (
                raw_geofence.geofence.lat_minmax[0]
                + raw_geofence.geofence.lat_minmax[1]
            )
            northern = hemi > 0
            derived_direction = True

        if zone_number_determined and derived_direction:
            return zone_number, northern, "geofence"

        # use the lon min & max to get the zone, and if only 1 is possible then it is resolved
        if not zone_number_determined:
            utm_min = utm.from_latlon(
                raw_geofence.geofence.lat_minmax[0], raw_geofence.geofence.lon_minmax[0]
            )
            utm_max = utm.from_latlon(
                raw_geofence.geofence.lat_minmax[1], raw_geofence.geofence.lon_minmax[1]
            )
            if utm_min[2] == utm_max[2]:
                zone_number = utm_min[2]
                if derived_direction:
                    return zone_number, northern, "geofence"

        # use the centre of the geofence as default
        centre_lat = (
            raw_geofence.geofence.lat_minmax[0] + raw_geofence.geofence.lat_minmax[1]
        ) / 2.0
        centre_lon = (
            raw_geofence.geofence.lon_minmax[0] + raw_geofence.geofence.lon_minmax[1]
        ) / 2.0
        centre_utm = utm.from_latlon(centre_lat, centre_lon)

        # default the zone to make sure coordinates can be parsed
        if not zone_number_determined:
            zone_number = centre_utm[2]
        if not derived_direction:
            northern = centre_lat > 0

        return zone_number, northern, UTM_ZONE_DEFAULT

    def _are_valid_utm_extractions(
        self, parsed_direction: List[Tuple[float, bool]]
    ) -> bool:
        # invalid if an easting point is over the easting limit
        for pd in parsed_direction:
            if not pd[1] and pd[0] >= EASTING_UPPER_LIMIT:
                return False

        return True

    def _is_scale(self, text: str) -> bool:
        text = text.lower()
        return any(t in text for t in ["scale", ":"])

    def _extract_utm(
        self,
        input: CoordinateInput,
        utm_zone: Tuple[int, bool, str],
        ocr_text_blocks_raw: DocTextExtraction,
        lonlat_results: Tuple[
            Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
        ],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        if not ocr_text_blocks_raw or len(ocr_text_blocks_raw.extractions) == 0:
            logger.info("WARNING! No ocr text blocks available!")
            return ({}, {})

        ocr_text_blocks = deepcopy(ocr_text_blocks_raw)
        for e in ocr_text_blocks.extractions:
            e.bounds = get_bounds_bounding_box(e.bounds)

        lon_clue = (
            lon_minmax[0] + lon_minmax[1]
        ) / 2  # mid-points of lon/lat hint area
        lat_clue = (
            lat_minmax[0] + lat_minmax[1]
        ) / 2  # (used below to determine if a extracted pt is lon or lat)

        lon_results, lat_results = lonlat_results

        # get utm coords and zone of clue coords
        # utm_clue = (easting, northing, zone number, zone letter)
        logger.info(f"lat clue: {lat_clue}\tlon clue: {lon_clue}")
        utm_clue = utm.from_latlon(lat_clue, lon_clue)
        easting_clue = utm_clue[0]
        northing_clue = utm_clue[1]

        idx = 0
        ne_matches: List[Tuple[int, Any, Tuple[int, int, int]]] = []
        parsed_directions: List[Tuple[float, bool]] = []
        for block in ocr_text_blocks.extractions:
            if not self._is_scale(block.text):
                matches_iter = RE_NORTHEAST.finditer(block.text.lower())
                for m in matches_iter:
                    m_groups = m.groups()
                    if any(x for x in m_groups):
                        # valid match
                        m_span = (m.start(), m.end(), len(block.text))

                        # if a non-default direction can be derived, store it for future use
                        utm_dist = RE_NONNUMERIC.sub("", m_groups[0])
                        utm_dist = float(utm_dist)
                        if utm_dist > 0:
                            is_northing = self._is_northing_point(
                                utm_dist, m_groups[0], m_groups[1], []
                            )
                            if not is_northing[1] == NORTHING_DEFAULT:
                                parsed_directions.append((utm_dist, is_northing[0]))

                        ne_matches.append((idx, m_groups, m_span))
            idx += 1

        valid_parsings = self._are_valid_utm_extractions(parsed_directions)

        # parse northing and easting points
        northings = []
        eastings = []
        for idx, groups, span in ne_matches:
            utm_dist = RE_NONNUMERIC.sub("", groups[0])
            utm_dist = float(utm_dist)
            if utm_dist == 0:  # skip noisy extraction
                continue

            if not valid_parsings:
                logger.info(
                    "Excluding candidate utm point due to invalid easting point: {}".format(
                        utm_dist
                    )
                )
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    "coordinate-excluded-utm",
                    {
                        "bounds": ocr_to_coordinates(
                            ocr_text_blocks.extractions[idx].bounds
                        ),
                        "text": ocr_text_blocks.extractions[idx].text,
                    },
                    "excluded due to some coordinates being over utm easting limit",
                )
                continue

            is_northing = self._is_northing_point(
                utm_dist, groups[0], groups[1], parsed_directions
            )

            if is_northing[0]:
                northings.append((utm_dist, span, idx))
            else:
                # longitude keypoint (x-axis)
                # given limited range of easting values and false easting values, use an arbitrary upper limit for upper bound
                if utm_dist < EASTING_UPPER_LIMIT:
                    eastings.append((utm_dist, span, idx))
                else:
                    logger.info(
                        "Excluding candidate easting point: {}".format(utm_dist)
                    )
        # convert from utm to lat/lon, using average values of northing and easting for the mapping coordinate
        # northing first as some may be excluded, then eastings
        if len(eastings) > 0:
            easting_clue = statistics.median(map(lambda x: x[0], eastings))
        for n, span, idx in northings:
            # latitude keypoint (y-axis)
            # check that it falls within the geofence by checking the lat absolute values
            # convert extracted northing value to latitude and save keypoint result
            latlon_pt = utm.to_latlon(
                easting_clue, n, utm_zone[0], northern=utm_zone[1]
            )
            # latlon_pt = (abs(latlon_pt[0]), abs(latlon_pt[1]))
            # if lat_minmax[0] <= latlon_pt[0] <= lat_minmax[1]:
            if is_in_range(latlon_pt[0], lat_minmax):
                # valid latitude point
                x_ranges = (
                    (0.0, 1.0)
                    if span[2] == 0
                    else (span[0] / float(span[2]), span[1] / float(span[2]))
                )
                coord = Coordinate(
                    "lat keypoint",
                    ocr_text_blocks.extractions[idx].text,
                    latlon_pt[0],
                    SOURCE_UTM,
                    True,
                    ocr_text_blocks.extractions[idx].bounds,
                    x_ranges=x_ranges,
                    confidence=0 if utm_zone[2] == UTM_ZONE_DEFAULT else 0.75,
                )
                x_pixel, y_pixel = coord.get_pixel_alignment()
                lat_results[(latlon_pt[0], y_pixel)] = coord
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    f"coordinate-{coord.get_type()}",
                    {
                        "bounds": ocr_to_coordinates(coord.get_bounds()),
                        "text": coord.get_text(),
                        "parsed": coord.get_parsed_degree(),
                        "type": "latitude" if coord.is_lat() else "longitude",
                        "pixel_alignment": coord.get_pixel_alignment(),
                        "confidence": coord.get_confidence(),
                    },
                    "extracted northing utm coordinate",
                )
            else:
                logger.info("Excluding candidate northing point: {}".format(n))

        if len(northings) > 0:
            northing_clue = statistics.median(map(lambda x: x[0], northings))
        for e, span, idx in eastings:
            x_ranges = (
                (0.0, 1.0)
                if span[2] == 0
                else (span[0] / float(span[2]), span[1] / float(span[2]))
            )
            # convert extracted easting value to longitude and save keypoint result
            latlon_pt = utm.to_latlon(
                e, northing_clue, utm_zone[0], northern=utm_zone[1]
            )
            # latlon_pt = (abs(latlon_pt[0]), abs(latlon_pt[1]))
            coord = Coordinate(
                "lon keypoint",
                ocr_text_blocks.extractions[idx].text,
                latlon_pt[1],
                SOURCE_UTM,
                False,
                ocr_text_blocks.extractions[idx].bounds,
                x_ranges=x_ranges,
                confidence=0 if utm_zone[2] == UTM_ZONE_DEFAULT else 0.75,
            )
            x_pixel, y_pixel = coord.get_pixel_alignment()
            lon_results[(latlon_pt[1], x_pixel)] = coord
            self._add_param(
                input.input,
                str(uuid.uuid4()),
                f"coordinate-{coord.get_type()}",
                {
                    "bounds": ocr_to_coordinates(coord.get_bounds()),
                    "text": coord.get_text(),
                    "parsed": coord.get_parsed_degree(),
                    "type": "latitude" if coord.is_lat() else "longitude",
                    "pixel_alignment": coord.get_pixel_alignment(),
                    "confidence": coord.get_confidence(),
                },
                "extracted easting utm coordinate",
            )

        logger.info("done utm")

        return (lon_results, lat_results)
