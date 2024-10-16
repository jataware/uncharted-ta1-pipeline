import copy
import csv
import logging
import re
import stateplane
import statistics
import uuid
import numpy as np

from shapely import Point as sPoint
from shapely.geometry import shape
from sklearn.cluster import DBSCAN
from copy import deepcopy

from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.text_extraction.entities import (
    DocTextExtraction,
    Point,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.entities import (
    CRS_OUTPUT_KEY,
    Coordinate,
    DocGeoFence,
    GEOFENCE_OUTPUT_KEY,
    SOURCE_STATE_PLANE,
    SOURCE_LAT_LON,
)
from tasks.geo_referencing.util import is_nad_83
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.util import (
    ocr_to_coordinates,
    get_bounds_bounding_box,
    is_in_range,
    get_min_max_count,
)

from util.json import read_json_file

from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("state_plane_extractor")

# UTM coordinates
# pre-compiled regex patterns
RE_NORTHEAST = re.compile(
    r"(?:^|\b)([1-9]?[ ]?[,.]?[ ]?[0-9]{3}[ ]?[,.]?[ ]?[0-9]00) ?m?(n|north|e|east)?($|\b)"
)

RE_NONNUMERIC = re.compile(r"[^0-9]")

FOV_RANGE_METERS = 200000  # fallback search range (meters)

UTM_ZONE_DEFAULT = "default"
NORTHING_DEFAULT = "default"
DIRECTION_DEFAULT = "default"

# FEET_FACTOR = 3.28083332904    # US feet
FEET_FACTOR = 3.280839895


class StatePlaneExtractor(CoordinatesExtractor):
    _code_lookup: Dict[Any, Any]

    def __init__(
        self,
        task_id: str,
        state_plane_lookup_filename: str,
        state_plane_zone_filename: str,
        state_code_filename: str,
    ):
        super().__init__(task_id)

        lookup_zones, lookup_fips = self._build_lookups(state_plane_lookup_filename)
        self._code_lookup = lookup_zones
        self._fips_lookup = lookup_fips

        self._zones = self._build_zone_lookup(state_plane_zone_filename)
        self._state_codes = self._build_state_code_lookup(state_code_filename)

    def _build_zone_lookup(
        self, state_plane_zone_filename: str
    ) -> List[Dict[Any, Any]]:
        data = read_json_file(state_plane_zone_filename)

        shapes = []
        for f in data["features"]:
            shapes.append({"shape": shape(f["geometry"]), "info": f["properties"]})
        return shapes

    def _build_state_code_lookup(self, state_code_filename: str) -> Dict[str, str]:
        # read the lookup file
        data = []
        with open(state_code_filename, newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        codes = {}
        for r in data[1:]:
            codes[r[0].lower()] = r[1].lower()
        return codes

    def _build_lookups(
        self, state_plane_lookup_filename: str
    ) -> Tuple[Dict[Any, Any], Dict[str, str]]:
        # read the lookup file
        data = []
        with open(state_plane_lookup_filename, newline="") as f:
            reader = csv.reader(f)
            data = list(reader)

        # reduce the file to projection -> state -> zones -> epsg codes
        # skip header
        lookup_zones = {"nad27": {}, "nad83": {}}
        lookup_fips = {}
        for r in data[1:]:
            state = r[1].lower()
            zone27 = r[4]
            zone83 = r[7]
            epsg27 = r[5]
            fips27 = r[6]
            epsg83 = r[8]
            if state not in lookup_zones["nad27"]:
                lookup_zones["nad27"][state] = {}
                lookup_zones["nad83"][state] = {}
            if zone27 not in lookup_zones["nad27"][state]:
                lookup_zones["nad27"][state][zone27] = epsg27
            if zone83 not in lookup_zones["nad83"][state]:
                lookup_zones["nad83"][state][zone83] = epsg83
            lookup_fips[fips27] = epsg27
        return lookup_zones, lookup_fips

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

        clue_point = input.input.get_request_info("clue_point")
        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")

        state_plane_zone = self._determine_epsg(
            metadata, geofence_raw, clue_point, lon_pts, lat_pts
        )
        if state_plane_zone[0] == "":
            logger.info("no state plane zone determined so stopping parsing attempt")
            # unable to determine state plane coordinates without a zone
            return lon_pts, lat_pts

        logger.info(f"derived state plane zone: {state_plane_zone}")

        lat_minmax = copy.deepcopy(geofence_raw.geofence.lat_minmax)
        lon_minmax = copy.deepcopy(geofence_raw.geofence.lon_minmax)

        lon_pts, lat_pts = self._extract_state_plane(
            input,
            state_plane_zone[0],
            ocr_blocks,
            metadata,
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
        north_east = list(map(lambda x: x[1], parsed_direction))
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

        return False, NORTHING_DEFAULT

    def _cluster_values(
        self, unparsed_values: List[Tuple[float, float, float]]
    ) -> Tuple[
        Optional[List[Tuple[float, float, float]]],
        Optional[List[Tuple[float, float, float]]],
    ]:
        # start with clustering approach
        # only need a few to cluster to determine lat and lon range
        data = np.array(list(map(lambda x: x[0], unparsed_values))).reshape(-1, 1)

        db = DBSCAN(eps=50000, min_samples=2).fit(data)
        labels = db.labels_

        # find the two biggest clusters
        data_clustered = list(zip(unparsed_values, labels))

        clusters = {}
        for d in data_clustered:
            if d[1] != -1:
                if d[1] not in clusters:
                    clusters[d[1]] = []
                clusters[d[1]].append(d[0])

        # assume lat & lon will be biggest 2 clusters
        cluster_list = []
        for _, c in clusters.items():
            cluster_list.append(c)

        # handle the case where fewer than 2 clusters are detected
        if len(cluster_list) < 1:
            return None, None
        elif len(cluster_list) == 1:
            return cluster_list[0], None

        cluster_list = sorted(cluster_list, key=lambda x: len(x), reverse=True)
        return cluster_list[0], cluster_list[1]

    def _split_directions(
        self, unparsed_values: List[Tuple[float, float, float]]
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]], str]:
        if len(unparsed_values) == 0:
            return [], [], DIRECTION_DEFAULT

        # find the two biggest clusters, assuming they are easting and northing
        c1, c2 = self._cluster_values(unparsed_values)
        if c1 is None:
            return [], [], DIRECTION_DEFAULT
        # handle the case with 1 main cluster and 1 left over coordinate
        if len(unparsed_values) - len(c1) == 1:
            c2 = list(set(unparsed_values) ^ set(c1))
        if c2 is None:
            return [], [], DIRECTION_DEFAULT

        # test if one set is easting
        is_easting, reason = self._is_x_direction(c1)
        if not reason == DIRECTION_DEFAULT:
            # is_easting indicates if c1 (true) is easting or c2 (false) is easting
            if is_easting:
                return c1, c2, reason
            else:
                return c2, c1, reason

        # if defaulted, test if other set is easting
        is_easting, reason = self._is_x_direction(c2)
        if not reason == DIRECTION_DEFAULT:
            # is_easting indicates if c2 (true) is easting or c1 (false) is easting
            if not is_easting:
                return c1, c2, reason
            else:
                return c2, c1, reason

        # if still defaulted, then return default
        return [], [], DIRECTION_DEFAULT

    def _determine_epsg_from_coord(
        self, projection: str, lon: float, lat: float, year: float
    ) -> str:
        if projection == "nad83" or year >= 1986:
            # can use the library to determine the epsg code as it uses NAD83
            return stateplane.identify(lon, lat)  #   type: ignore

        # figure out which zone the coordinate falls within
        point = sPoint(lon, lat)
        for z in self._zones:
            if z["shape"].contains(point):
                # use the fips code to lookup the epsg with the fips code being at least 4 characters
                fips: str = z["info"]["FIPSZONE"]
                fips = fips.rjust(4, "0")
                return self._fips_lookup[fips]

        # no zone info available, possibly a territory or different country
        return ""

    def _determine_epsg(
        self,
        metadata: MetadataExtraction,
        raw_geofence: DocGeoFence,
        clue_point: Optional[Tuple[float, float]],
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> Tuple[str, str]:
        logger.info("attempting to determine state plane zone")
        year = 1900
        if metadata.year.isdigit():
            year = int(metadata.year)

        # determine nad27 vs nad83
        projection = "nad83" if is_nad_83(metadata) else "nad27"

        # use clue point if available
        if clue_point is not None:
            return (
                self._determine_epsg_from_coord(
                    projection, clue_point[0], clue_point[1], year
                ),
                "clue point",
            )

        # use middle of parsed lon & lat if some of both exist and they fall within geofence
        centre_lat = (
            raw_geofence.geofence.lat_minmax[0] + raw_geofence.geofence.lat_minmax[1]
        ) / 2
        centre_lon = (
            raw_geofence.geofence.lon_minmax[0] + raw_geofence.geofence.lon_minmax[1]
        ) / 2
        min_lon, max_lon, count_lon = get_min_max_count(
            lons, centre_lon < 0, [SOURCE_LAT_LON]
        )
        min_lat, max_lat, count_lat = get_min_max_count(
            lats, centre_lat < 0, [SOURCE_LAT_LON]
        )
        if count_lon > 0 and count_lat > 0:
            return (
                self._determine_epsg_from_coord(
                    projection, (min_lon + max_lon) / 2, (min_lat + max_lat) / 2, year
                ),
                "parsed coordinates",
            )

        # use the state from the metadata
        # TODO: FIGURE OUT WHICH OF THE POSSIBLE STATES TO USE
        states = [s for s in metadata.states if not s == "NULL"]
        if len(states) > 0:
            state = states[0].lower()
            logger.debug(f"narrowing state plane zone to state {state}")
            state_code = self._get_state_code(state)
            logger.debug(f"narrowing state plane zone to state code {state_code}")
            if state_code in self._code_lookup[projection]:
                possible = self._code_lookup[projection][state_code]
                if len(possible) == 1:
                    # only one zone exists in the state
                    return list(possible.items())[0][1], "only option"

                # TODO: check if the parsed coordinates can narrow down the options

                # use the projection info to try and narrow it down to one zone
                for n, c in possible.items():
                    if n.lower() in list(
                        map(lambda x: x.lower(), metadata.coordinate_systems)
                    ):
                        return c, CRS_OUTPUT_KEY

        # use the centre of the geofence or parsed coordinates to pick the code
        if count_lon != 0:
            centre_lon = (min_lon + max_lon) / 2
        if count_lat != 0:
            centre_lat = (min_lat + max_lat) / 2
        return (
            self._determine_epsg_from_coord(projection, centre_lon, centre_lat, year),
            "default",
        )  #   type: ignore

    def _get_state_code(self, state: str) -> str:
        # depending on parsed metadata, could either be 'US-STATE CODE' or STATE
        if len(state) == 5 and state.startswith("us"):
            return state[-2:]
        return self._state_codes[state]

    def _is_scale(self, text: str) -> bool:
        text = text.lower()
        return any(t in text for t in ["scale", ":"])

    def _requires_meters(self, metadata: MetadataExtraction) -> bool:
        # TODO: FIGURE OUT HOW THIS COULD BE DERIVED AND IF NECESSARY
        # nad27 maps require feet for projections, nad83 maps do not
        return is_nad_83(metadata)

    def _extract_state_plane(
        self,
        input: CoordinateInput,
        state_plane_zone: str,
        ocr_text_blocks_raw: DocTextExtraction,
        metadata: MetadataExtraction,
        lonlat_results: Tuple[
            Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
        ],
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        if not ocr_text_blocks_raw or len(ocr_text_blocks_raw.extractions) == 0:
            logger.warning("No ocr text blocks available!")
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
        logger.debug(f"lat clue: {lat_clue}\tlon clue: {lon_clue}")
        utm_clue = stateplane.from_latlon(lat_clue, lon_clue)
        easting_clue = utm_clue[0]
        northing_clue = utm_clue[1]

        lon_results, lat_results = lonlat_results

        idx = 0
        ne_matches: List[Tuple[int, Any, Tuple[int, int, int]]] = []
        parsed_directions: List[Tuple[float, bool]] = []
        unparsed_directions: List[Tuple[float, float, float]] = []
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

                        # need to work with meters
                        if self._requires_meters(metadata):
                            utm_dist = utm_dist / FEET_FACTOR

                        if utm_dist > 0:
                            is_northing = self._is_northing_point(
                                utm_dist, m_groups[0], m_groups[1], []
                            )
                            if not is_northing[1] == NORTHING_DEFAULT:
                                parsed_directions.append((utm_dist, is_northing[0]))
                            else:
                                centers = self._get_bounds_center(block.bounds)
                                unparsed_directions.append(
                                    (utm_dist, centers[0], centers[1])
                                )

                        ne_matches.append((idx, m_groups, m_span))
            idx += 1
        if len(parsed_directions) == 0:
            # determine direction based on unparsed directions
            easting_set, northing_set, reason = self._split_directions(
                unparsed_directions
            )
            if not reason == DIRECTION_DEFAULT:
                logger.info(f"split easting and northing by using {reason}")
                for e in easting_set:
                    parsed_directions.append((e[0], False))
                for n in northing_set:
                    parsed_directions.append((n[0], True))

        # parse northing and easting points
        northings = []
        eastings = []
        for idx, groups, span in ne_matches:
            utm_dist = RE_NONNUMERIC.sub("", groups[0])
            utm_dist = float(utm_dist)
            if utm_dist == 0:  # skip noisy extraction
                continue

            # need to work with meters
            if self._requires_meters(metadata):
                utm_dist = utm_dist / FEET_FACTOR

            is_northing = self._is_northing_point(
                utm_dist, groups[0], groups[1], parsed_directions
            )

            if is_northing[0]:
                northings.append((utm_dist, span, idx))
            else:
                eastings.append((utm_dist, span, idx))

        # convert from utm to lat/lon, using average values of northing and easting for the mapping coordinate
        # northing first as some may be excluded, then eastings
        if len(eastings) > 0:
            easting_clue = statistics.median(map(lambda x: x[0], eastings))
        for n, span, idx in northings:
            # latitude keypoint (y-axis)
            # check that it falls within the geofence by checking the lat absolute values
            # convert extracted northing value to latitude and save keypoint result
            latlon_pt = stateplane.to_latlon(easting_clue, n, epsg=state_plane_zone)
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
                    SOURCE_STATE_PLANE,
                    True,
                    ocr_text_blocks.extractions[idx].bounds,
                    x_ranges=x_ranges,
                    confidence=0.75,
                )
                x_pixel, y_pixel = coord.get_pixel_alignment()
                lat_results[(latlon_pt[0], y_pixel)] = coord

            else:
                logger.debug("Excluding candidate northing point: {}".format(n))

        if len(northings) > 0:
            northing_clue = statistics.median(map(lambda x: x[0], northings))
        for e, span, idx in eastings:
            x_ranges = (
                (0.0, 1.0)
                if span[2] == 0
                else (span[0] / float(span[2]), span[1] / float(span[2]))
            )
            # convert extracted easting value to longitude and save keypoint result
            latlon_pt = stateplane.to_latlon(e, northing_clue, epsg=state_plane_zone)
            # latlon_pt = (abs(latlon_pt[0]), abs(latlon_pt[1]))
            coord = Coordinate(
                "lon keypoint",
                ocr_text_blocks.extractions[idx].text,
                latlon_pt[1],
                SOURCE_STATE_PLANE,
                False,
                ocr_text_blocks.extractions[idx].bounds,
                x_ranges=x_ranges,
                confidence=0.75,
            )
            x_pixel, y_pixel = coord.get_pixel_alignment()
            lon_results[(latlon_pt[1], x_pixel)] = coord

        return (lon_results, lat_results)

    def _get_bounds_center(self, bounds: List[Point]) -> Tuple[float, float]:
        # get the min & max for x & y
        x_values = list(map(lambda x: x.x, bounds))
        y_values = list(map(lambda x: x.y, bounds))

        # use the middle of the range for the center
        # TODO: check if bounds is a polygon and perhaps derive the middle differently
        return (
            (min(x_values) + max(x_values)) / 2,
            (min(y_values) + max(y_values)) / 2,
        )

    def _is_x_direction(
        self, values: List[Tuple[float, float, float]]
    ) -> Tuple[bool, str]:
        # if the same parsed value is seen multiple times, determine if x is the main changing direction
        values_lookup = {}
        for v in values:
            if v[0] in values_lookup:
                # two identical values, the direction with the smallest change defines the direction of the axis
                first = values_lookup[v[0]]
                x_delta = abs(first[1] - v[1])
                y_delta = abs(first[2] - v[2])
                return x_delta < y_delta, "overlap"
            values_lookup[v[0]] = v

        # all points are unique values, but can still be along different sides of a map for the same direction
        if len(values) > 2:
            # can determine direction in most cases as at least 2 will be on the same line
            # polygons may fall outside of that assumption
            # sort values by x and y
            values_x = copy.deepcopy(values)
            values_x.sort(key=lambda x: x[1])

            values_y = copy.deepcopy(values)
            values_y.sort(key=lambda x: x[2])

            # find the smallest differences by x and y
            x_index = self._find_smallest_difference(
                list(map(lambda x: x[1], values_x))
            )
            y_index = self._find_smallest_difference(
                list(map(lambda x: x[2], values_y))
            )

            # calculate the change by unit of value
            x_pixel_diff = values_x[x_index + 1][1] - values_x[x_index][1]
            x_value_diff = values_x[x_index + 1][0] - values_x[x_index][0]
            x_ratio = x_pixel_diff / x_value_diff

            # ratio of x to y differences should determine direction
            y_pixel_diff = values_y[y_index + 1][1] - values_y[y_index][1]
            y_value_diff = values_y[y_index + 1][0] - values_y[y_index][0]
            y_ratio = y_pixel_diff / y_value_diff
            return x_ratio < y_ratio, "variance"

        # if 2 points but only significant variance in one axis, then the changing axis identifies the direction
        if len(values) == 2:
            x_delta = abs(values[0][1] - values[1][1])
            y_delta = abs(values[0][2] - values[1][2])
            ratio = float(x_delta) / y_delta
            if ratio > 5 or ratio < 0.2:
                return x_delta > y_delta, "delta"

        # if 2 points with significant change across both axes or fewer than 2 points, then default
        return True, DIRECTION_DEFAULT

    def _find_smallest_difference(self, values: List[float]) -> int:
        # should have at least 3 values, and the list should be sorted
        diff = values[1] - values[0]
        i_s = 0
        for i in range(len(values) - 1):
            diff_c = values[i + 1] - values[i]
            if diff_c < diff:
                i_s = i

        return i_s
