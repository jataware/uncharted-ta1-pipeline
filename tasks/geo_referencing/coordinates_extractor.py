import logging
import re
from copy import deepcopy

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.text_extraction.entities import (
    DocTextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.entities import (
    Coordinate,
    CoordType,
    CoordSource,
)
from tasks.geo_referencing.geo_coordinates import split_lon_lat_degrees
from tasks.geo_referencing.util import (
    get_bounds_bounding_box,
    get_input_geofence,
)
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("coordinates_extractor")

# GeoCoordinates
# pre-compiled regex patterns
RE_DMS = re.compile(
    r"^|\b[-+]?([0-9]{1,2}|1[0-7][0-9]|180)( |[o*°⁰˙˚•·º:]|( ?,?[o*°⁰˙˚•·º:]))( ?[0-6]?[0-9])['`′/:]?( ?[0-6][0-9])?[\"″'`′/:]?($|\b)"
)  # match degrees with minutes (and optionally seconds)

RE_DEG = re.compile(
    r"^|\b[-+]?([0-9]{1,2}|1[0-7][0-9]|180) ?,?[o*°⁰˙˚•·º⁹]($|\b)"
)  # match degrees only

RE_DEGDEC = re.compile(
    r"^|\b[-+]?([0-9]{1,2}|1[0-7][0-9]|180)(\.[0-9]{1,3})[o*°⁰˙˚•·º]?($|\b)"
)  # match degrees with decimals only

RE_DEGMIN = re.compile(
    r"^|\b([0-9]{1,2}|1[0-7][0-9]|180) |['`′/](00|15|30|45)[\"″'`′/]?($|\b)"
)  # coarse match of degrees + minutes

RE_MINSSECS = re.compile(
    r"^|\b([0-6][0-9])['`′/]( ?[0-6][0-9])?[\"″'`′/]?($|\b)"
)  # match minutes only (and optionally seconds)

RE_NONNUMERIC = re.compile(r"[^0-9]")

RE_LETTERS = re.compile(r"[a-zA-Z]+")  # matches one or more letters

MINUTES_PASSLIST = [0, 15, 30, 45]

MIN_PLACENAME_LEN = 3

# Geocode coordinates
GEOCODE_CACHE = "temp/geocode/"

# from Google Geocoder API -- see https://developers.google.com/maps/documentation/geocoding/requests-geocoding#Types
# LOCATION_TYPES_BLOCKLIST = ['country', 'administrative_area_level_1', 'administrative_area_level_2']
LOC_TYPES_PASSLIST_GOOGLE = ["locality", "sublocality", "sublocality_level_1"]

# from Nominatim / Openstreetmap ontology
# https://wiki.openstreetmap.org/wiki/Map_features#Place
LOC_TYPES_PASSLIST_NOM = [
    "municipality",
    "city",
    "borough",
    "town",
    "village",
    "hamlet",
    "locality",
]


class ParsedDegree:
    degree: float
    minutes: float
    seconds: float


class CoordinateInput:
    input: TaskInput
    updated_output: Dict[Any, Any] = {}

    def __init__(self, input: TaskInput):
        self.input = input
        self.updated_output = {}


class CoordinatesExtractor(Task):
    def run(self, input: TaskInput) -> TaskResult:

        input_coord = CoordinateInput(input)

        if not self._should_run(input_coord):
            return self._create_result(input_coord.input)

        # extract the coordinates using the input
        lats = input.get_data("lats", [])
        lons = input.get_data("lons", [])
        lons, lats = self._extract_coordinates(input_coord)
        logger.info(
            f"Num coordinates extracted: {len(lats)} latitude and {len(lons)} longitude"
        )

        # add the extracted coordinates to the result
        return self._create_coordinate_result(input_coord, lons, lats)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return {}, {}

    def _get_input_geofence(
        self, input: CoordinateInput
    ) -> Tuple[List[float], List[float], bool]:
        return get_input_geofence(input.input)

    def _create_coordinate_result(
        self,
        input: CoordinateInput,
        lons: Dict[Tuple[float, float], Coordinate],
        lats: Dict[Tuple[float, float], Coordinate],
    ) -> TaskResult:
        result = super()._create_result(input.input)

        result.output["lons"] = lons
        result.output["lats"] = lats

        for k, v in input.updated_output.items():
            result.output[k] = v

        return result

    def _should_run(self, input: CoordinateInput) -> bool:
        lats = input.input.get_data("lats", {})
        lons = input.input.get_data("lons", {})
        num_keypoints = min(len(lons), len(lats))
        return num_keypoints < 2

        # TODO: could check the number of lats and lons with status == OK
        # lats = list(filter(lambda c: c._status == CoordStatus.OK, lats.values()))
        # lons = list(filter(lambda c: c._status == CoordStatus.OK, lons.values()))

class GeoCoordinatesExtractor(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        ocr_blocks: DocTextExtraction = input.input.parse_data(
            TEXT_EXTRACTION_OUTPUT_KEY, DocTextExtraction.model_validate
        )
        lon_minmax, lat_minmax, defaulted = self._get_input_geofence(input)

        lon_pts, lat_pts = self._extract_lonlat(
            input, ocr_blocks, lon_minmax, lat_minmax, defaulted
        )

        return lon_pts, lat_pts

    def _parse_groups(self, groups: List[str]) -> Optional[ParsedDegree]:
        parsed = ParsedDegree()
        try:
            deg = RE_NONNUMERIC.sub("", groups[0])
            deg = float(deg)
        except:
            return None  # not valid

        try:
            minutes = RE_NONNUMERIC.sub("", groups[3])
            if minutes:
                minutes = float(minutes)
            else:
                minutes = 0.0
        except:
            minutes = 0.0
        try:
            seconds = RE_NONNUMERIC.sub("", groups[4])
            if seconds:
                seconds = float(seconds)
            else:
                seconds = 0.0
        except:
            seconds = 0.0

        parsed.degree = deg
        parsed.minutes = minutes
        parsed.seconds = seconds
        return parsed

    def _extract_lonlat(
        self,
        input: CoordinateInput,
        ocr_text_blocks_raw: DocTextExtraction,
        lon_minmax: List[float],
        lat_minmax: List[float],
        default_geofence: bool,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        ocr_text_blocks = deepcopy(ocr_text_blocks_raw)
        for e in ocr_text_blocks.extractions:
            e.bounds = get_bounds_bounding_box(e.bounds)

        lon_clue = (
            lon_minmax[0] + lon_minmax[1]
        ) / 2  # mid-points of lon/lat hint area
        lat_clue = (
            lat_minmax[0] + lat_minmax[1]
        ) / 2  # (used below to determine if a extracted pt is lon or lat)

        idx = 0
        dms_matches: Dict[int, Tuple[Any, Tuple[int, int, int], float]] = {}
        deg_matches: Dict[int, Tuple[Any, Tuple[int, int, int]]] = {}
        degmin_matches: Dict[int, Tuple[Any, Tuple[int, int, int]]] = {}
        minsec_matches = {}
        for block in ocr_text_blocks.extractions:
            is_match = False
            matches_iter = RE_DMS.finditer(block.text)
            for m in matches_iter:
                m_groups = m.groups()
                if any(x for x in m_groups):
                    # valid match
                    is_match = True
                    m_span = (m.start(), m.end(), len(block.text))
                    ok, deg_parsed = self._parse_dms(m_groups)
                    if ok:
                        parsed_dms = self._parse_groups(m_groups)  #   type:ignore
                        if parsed_dms:
                            # minutes and seconds expected in certain increments
                            if self._check_degree_increments(
                                int(parsed_dms.minutes), int(parsed_dms.seconds)
                            ):
                                dms_matches[idx] = (m_groups, m_span, deg_parsed)
                            else:
                                logger.debug(
                                    f"Excluding candidate point due to unexpected degree increment: {deg_parsed}"
                                )
                        else:
                            dms_matches[idx] = (m_groups, m_span, deg_parsed)
            if not is_match:
                matches_iter = RE_DEGMIN.finditer(block.text)
                for m in matches_iter:
                    m_groups = m.groups()
                    if any(x for x in m_groups):
                        # valid match
                        is_match = True
                        m_span = (m.start(), m.end(), len(block.text))
                        degmin_matches[idx] = (m_groups, m_span)
            if not is_match:
                matches_iter = RE_DEG.finditer(block.text)
                for m in matches_iter:
                    m_groups = m.groups()
                    if not m_groups[0] or m_groups[0].startswith("0"):
                        continue
                    if any(x for x in m_groups):
                        # valid match
                        is_match = True
                        m_span = (m.start(), m.end(), len(block.text))
                        deg_matches[idx] = (m_groups, m_span)
            if not is_match:
                matches_iter = RE_DEGDEC.finditer(block.text)
                for m in matches_iter:
                    m_groups = m.groups()
                    if any(x for x in m_groups):
                        # valid match
                        is_match = True
                        m_span = (m.start(), m.end(), len(block.text))
                        deg_matches[idx] = (
                            m_groups,
                            m_span,
                        )  # put in same dict as deg_matches
            if not is_match:
                matches_iter = RE_MINSSECS.finditer(block.text)
                for m in matches_iter:
                    m_groups = m.groups()
                    if any(x for x in m_groups):
                        # valid match
                        is_match = True
                        m_span = (m.start(), m.end(), len(block.text))
                        minsec_matches[idx] = (m_groups, m_span)
            idx += 1

        if default_geofence:
            updated_geofence = self._get_geofence(
                [lon_minmax, lat_minmax], dms_matches, ocr_text_blocks
            )
            # the updated geofence should only narrow the existing one so make sure it falls within the initial geofence
            if (
                updated_geofence[0][0] >= lon_minmax[0]
                and updated_geofence[0][1] <= lon_minmax[1]
            ):
                lon_minmax = updated_geofence[0]
                input.updated_output["lon_minmax"] = lon_minmax
                lon_clue = (lon_minmax[0] + lon_minmax[1]) / 2
            if (
                updated_geofence[1][0] >= lat_minmax[0]
                and updated_geofence[1][1] <= lat_minmax[1]
            ):
                lat_minmax = updated_geofence[1]
                input.updated_output["lat_minmax"] = lat_minmax
                lat_clue = (lat_minmax[0] + lat_minmax[1]) / 2
            logger.info(
                f"New geo fence: {updated_geofence}\tlon clue: {lon_clue}\tlat clue: {lat_clue}"
            )

        # ---- finalize lat/lon results
        coord_results = []
        coord_lat_results = {}
        coord_lon_results = {}
        # ---- Check degress-minutes-seconds extractions...
        for idx, (groups, span, _) in dms_matches.items():
            parsed_dms = self._parse_groups(groups)
            if parsed_dms is None:
                continue

            # convert DMS to decimal degrees
            deg_decimal = (
                parsed_dms.degree
                + parsed_dms.minutes / 60.0
                + parsed_dms.seconds / 3600.0
            )

            if self._check_consecutive(
                parsed_dms.degree, parsed_dms.minutes, parsed_dms.seconds
            ):
                logger.debug("Excluding candidate point: {}".format(deg_decimal))
                continue

            # could fit in only one geo fence
            is_lat = (lat_minmax[0] <= deg_decimal <= lat_minmax[1]) and not (
                lon_minmax[0] <= deg_decimal <= lon_minmax[1]
            )
            is_lon = not (lat_minmax[0] <= deg_decimal <= lat_minmax[1]) and (
                lon_minmax[0] <= deg_decimal <= lon_minmax[1]
            )

            if is_lat or (
                not is_lon and abs(deg_decimal - lat_clue) < abs(deg_decimal - lon_clue)
            ):
                # latitude keypoint (y-axis)
                if deg_decimal >= lat_minmax[0] and deg_decimal <= lat_minmax[1]:
                    # valid latitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        True,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.95,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    # pixel->latitude mapping depends mostly on y-pixel (but also x-pixel values, due to possible map rotation/projection)
                    coord_lat_results[(deg_decimal, y_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate latitude point: {} with lat minmax {}".format(
                            deg_decimal, lat_minmax
                        )
                    )
            else:
                # longitude keypoint (x-axis)
                if deg_decimal >= lon_minmax[0] and deg_decimal <= lon_minmax[1]:
                    # valid longitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        False,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.95,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    # pixel->longitude mapping depends mostly on x-pixel (but also y-pixel values, due to possible map rotation/projection)
                    coord_lon_results[(deg_decimal, x_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate longitude point: {}".format(deg_decimal)
                    )

        # ---- Check degrees-minutes extractions... (potentially more coarse/noisy)
        for idx, (groups, span) in degmin_matches.items():
            try:
                deg = RE_NONNUMERIC.sub("", groups[0])
                deg = float(deg)
                minutes = RE_NONNUMERIC.sub("", groups[1])
                if minutes:
                    minutes = float(minutes)
                else:
                    minutes = 0.0
            except:
                continue  # not valid

            # convert DMS to decimal degrees
            deg_decimal = deg + minutes / 60.0

            if self._check_consecutive(deg, minutes, 0):
                logger.debug("Excluding candidate point: {}".format(deg_decimal))
                continue

            if abs(deg_decimal - lat_clue) < abs(deg_decimal - lon_clue):
                # latitude keypoint (y-axis)
                if deg_decimal >= lat_minmax[0] and deg_decimal <= lat_minmax[1]:
                    # valid latitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        True,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.9,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    # pixel->latitude mapping depends mostly on y-pixel (but also x-pixel values, due to possible map rotation/projection)
                    coord_lat_results[(deg_decimal, y_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate latitude point: {}".format(deg_decimal)
                    )
            else:
                # longitude keypoint (x-axis)
                if deg_decimal >= lon_minmax[0] and deg_decimal <= lon_minmax[1]:
                    # valid longitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        False,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.9,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    # pixel->longitude mapping depends mostly on x-pixel (but also y-pixel values, due to possible map rotation/projection)
                    coord_lon_results[(deg_decimal, x_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate longitude point: {}".format(deg_decimal)
                    )

        # ideally, we should have 3 or more keypoints for latitude and longitude each
        # to estimate map rotation, scale and translation tranformations
        if len(coord_lat_results) >= 3 and len(coord_lon_results) >= 3:
            return (coord_lon_results, coord_lat_results)

        # -----
        # try to also parse degrees (potentially without minutes/seconds; more coarse result)
        for idx, (groups, span) in deg_matches.items():
            try:
                deg = RE_NONNUMERIC.sub("", groups[0])
            except:
                continue
            is_decimal = False
            if len(groups) > 1 and groups[1]:
                is_decimal = True
                deg += groups[1]  # add decimal part of degrees
            deg = float(deg)

            minutes = 0.0
            seconds = 0.0
            font_height = 0.0
            if idx + 1 in minsec_matches and not is_decimal:
                # perhaps a corresponding minutes/seconds OCR block is available (next block after this degrees text)?
                # also, check relative spacing between the two OCR groups (is one right below the other?)
                font_height = (
                    ocr_text_blocks.extractions[idx].bounds[2].y
                    - ocr_text_blocks.extractions[idx].bounds[0].y
                )
                blocks_y_separation = (
                    ocr_text_blocks.extractions[idx + 1].bounds[0].y
                    - ocr_text_blocks.extractions[idx].bounds[2].y
                )

                if blocks_y_separation < font_height:
                    # these successive text blocks are close together (y-axis), so they they might be related...
                    minsec_groups, minsec_span = minsec_matches[idx + 1]
                    try:
                        minutes = RE_NONNUMERIC.sub("", minsec_groups[0])
                        if minutes:
                            minutes = float(minutes)
                        else:
                            minutes = 0.0
                    except:
                        minutes = 0.0
                    try:
                        seconds = RE_NONNUMERIC.sub("", minsec_groups[1])
                        if seconds:
                            seconds = float(seconds)
                        else:
                            seconds = 0.0
                    except:
                        seconds = 0.0

            # convert DMS to decimal degrees
            deg_decimal = deg + minutes / 60.0 + seconds / 3600.0

            if abs(deg_decimal - lat_clue) < abs(deg_decimal - lon_clue):
                # latitude keypoint (y-axis)
                if deg_decimal >= lat_minmax[0] and deg_decimal <= lat_minmax[1]:
                    # valid latitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        True,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        font_height=font_height,
                        confidence=0.8,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    coord_lat_results[(deg_decimal, y_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate latitude point: {}".format(deg_decimal)
                    )
            else:
                # longitude keypoint (x-axis)
                if deg_decimal >= lon_minmax[0] and deg_decimal <= lon_minmax[1]:
                    # valid longitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    coord = Coordinate(
                        CoordType.KEYPOINT,
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
                        CoordSource.LAT_LON,
                        False,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        font_height=font_height,
                        confidence=0.8,
                    )
                    coord_results.append(coord)
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    coord_lon_results[(deg_decimal, x_pixel)] = coord
                else:
                    logger.debug(
                        "Excluding candidate longitude point: {}".format(deg_decimal)
                    )

        return (coord_lon_results, coord_lat_results)

    def _check_degree_increments(self, minutes: int, seconds: int) -> bool:
        # support 1/16 or 1/12 degree increments
        # disabled for now as inner roi targets the same issue
        # TODO: coordinate this function with inner ROI filtering somehow
        return (
            (minutes * 60 + seconds) % 225 == 0
            or (minutes * 60 + seconds) % 300 == 0
            or True
        )

    def _check_consecutive(self, deg: float, minutes: float, seconds: float) -> bool:
        # checks if lat/lon extraction is a series of consecutive numbers
        # eg 49, 50, 51
        is_consecutive = False
        if deg == 0 or minutes == 0:
            return False
        if minutes in [0, 15, 30, 45]:
            return False
        if abs(deg - minutes) == 1:
            is_consecutive = True
        if is_consecutive and seconds != 0:
            is_consecutive = abs(minutes - seconds) == 1
        return is_consecutive

    def _remove_outlier_pts(
        self,
        input: CoordinateInput,
        deg_results: Dict[Tuple[float, float], Coordinate],
        i_max: int,
        j_max: int,
    ):
        deg_results_mapped = {}
        for k, v in deg_results.items():
            deg_results_mapped[k] = v.to_deg_result()[1]

        deg_groups = []
        max_group_size = 3
        i_delta = 0.05 * i_max  # 5 percent
        j_delta = 0.05 * j_max
        # loop through all keypoints and group
        # based on x,y approx alignment
        for (deg, pxl_i), pxl_j in deg_results_mapped.items():
            in_group = False
            for this_group in deg_groups:
                if len(this_group) >= max_group_size:
                    continue
                (g_deg, g_i), g_j = list(this_group.items())[0]
                if abs(pxl_i - g_i) < i_delta and deg == g_deg:
                    in_group = True
                elif abs(pxl_j - g_j) < j_delta:
                    # NOTE: lon or lat values assumed to have -ve slope
                    # wrt pixel value (assumes N. America geo-location)
                    if deg > g_deg and pxl_i < g_i:
                        in_group = True
                    elif deg < g_deg and pxl_i > g_i:
                        in_group = True
                if in_group:
                    this_group[(deg, pxl_i)] = pxl_j
                    break

            if not in_group:
                # make a new group
                new_g = {}
                new_g[(deg, pxl_i)] = pxl_j
                deg_groups.append(new_g)

        # keep keypoints from the 'best' candidate group
        # based on group size and overall pixel span
        span_max = 0
        idx_span = 0
        in_group = False
        for idx, this_group in enumerate(deg_groups):
            if len(this_group) > 1:
                i_vals = [g_i for (g_deg, g_i), g_j in this_group.items()]
                deg_vals = [g_deg for (g_deg, g_i), g_j in this_group.items()]
                deg_vals = set(deg_vals)  # how many unique deg values in this group?
                i_span = (max(i_vals) - min(i_vals)) * len(
                    this_group
                )  # max pxl span (boosted by group size)
                if i_span > span_max and len(deg_vals) > 1:
                    span_max = i_span
                    idx_span = idx
                    in_group = True
        if in_group:
            # flag removed outliers
            deg_results_clean = {}
            for k, v in deg_results.items():
                if k in deg_groups[idx_span]:
                    deg_results_clean[k] = deg_results[k]
            return deg_results_clean

        return deg_results

    def _filter_secondary_map(
        self, coord_result: Dict[Tuple[float, float], Coordinate]
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        # remove lat & lon points that are part of a secondary map
        if len(coord_result) == 0:
            return coord_result, {}

        # determine the min & max lon and lat values
        min_coord, max_coord = min(coord_result, key=lambda x: x[1]), max(
            coord_result, key=lambda x: x[1]
        )

        # flag points whose pixel coordinate change from min don't at all match expectations given the range
        # they could be negative (which would imply a secondary map coordinate) or increasing way too fast
        # TODO: need to handle skewed maps somehow
        coord_filtered = {}
        coord_dropped = {}
        if min_coord != max_coord:
            # some may be outliers due to being a secondary map or similar reason
            # use the widest range possible to determine the ratio of pixels to degrees
            delta_deg = min_coord[0] - max_coord[0]
            if delta_deg == 0:
                return coord_result, {}

            delta_pixel = max_coord[1] - min_coord[1]
            delta_pixel_deg = delta_pixel / delta_deg
            logger.debug(f"min coord: {min_coord}\tmax coord: {max_coord}")
            logger.debug(
                f"delta pixel: {delta_pixel}\tdelta deg:{delta_deg}\tratio: {delta_pixel_deg}"
            )
            for r in coord_result:
                delta_deg_pt = min_coord[0] - r[0]
                if delta_deg_pt == 0:
                    coord_filtered[r] = coord_result[r]
                    continue

                delta_pixel_pt = r[1] - min_coord[1]
                delta_ratio = (delta_pixel_pt / delta_deg_pt) / delta_pixel_deg

                # arbitrary limit for expected ratio of pixels to degrees
                if delta_ratio > 0.5:
                    coord_filtered[r] = coord_result[r]
                else:
                    logger.debug(f"dropping {r} due to being part of secondary map")
                    coord_dropped[r] = coord_result[r]
        return coord_filtered, coord_dropped

    def _get_geofence(
        self,
        geofence: List[List[float]],
        dms_matches: Dict[int, Tuple[Any, Tuple[int, int, int], float]],
        ocr_text_blocks: DocTextExtraction,
    ) -> List[List[float]]:
        degrees = []
        for idx, (_, _, deg_parsed) in dms_matches.items():
            # reduce parsed value to middle point
            bounding_box = ocr_text_blocks.extractions[idx].bounds
            (x, y) = (bounding_box[0].x + bounding_box[2].x) / 2, (
                bounding_box[0].y + bounding_box[2].y
            ) / 2
            degrees.append((x, y, deg_parsed))
        updated_geofence = split_lon_lat_degrees(geofence, degrees)

        return updated_geofence

    def _parse_dms(self, groups: Any) -> Tuple[bool, float]:
        try:
            deg = RE_NONNUMERIC.sub("", groups[0])
            deg = float(deg)
        except:
            return False, 0  # not valid

        try:
            minutes = RE_NONNUMERIC.sub("", groups[3])
            if minutes:
                minutes = float(minutes)
            else:
                minutes = 0.0
        except:
            minutes = 0.0
        try:
            seconds = RE_NONNUMERIC.sub("", groups[4])
            if seconds:
                seconds = float(seconds)
            else:
                seconds = 0.0
        except:
            seconds = 0.0

        # convert DMS to decimal degrees
        return True, deg + minutes / 60.0 + seconds / 3600.0
