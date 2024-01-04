import os
import re
import uuid

from copy import deepcopy
from geopy.geocoders import Nominatim, GoogleV3
from geopy.point import Point
import matplotlib.path as mpltPath

from tasks.common.task import Task, TaskInput, TaskResult
from tasks.text_extraction.entities import (
    DocTextExtraction,
    Point as TPoint,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.entities import Coordinate
from tasks.geo_referencing.geo_coordinates import split_lon_lat_degrees
from tasks.geo_referencing.util import ocr_to_coordinates, get_bounds_bounding_box
from util.cache import cache_geocode_data, load_geocode_cache

from typing import Any, Dict, List, Tuple

# GeoCoordinates
# pre-compiled regex patterns
RE_DMS = re.compile(
    r"^|\b[-+]?([0-9]{1,2}|1[0-7][0-9]|180)( |[o*°⁰˙˚•·º:]|( ?,?[o*°⁰˙˚•·º:]))( ?[0-6]?[0-9])['`′/]?( ?[0-6][0-9])?[\"″'`′/]?($|\b)"
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


class CoordinateInput:
    input: TaskInput
    updated_output: Dict[Any, Any] = {}

    def __init__(self, input: TaskInput):
        self.input = input
        self.updated_output = {}


class CoordinatesExtractor(Task):
    def run(self, input: TaskInput) -> TaskResult:
        print(f"running coordinates extraction task with id {self._task_id}")
        input_coord = CoordinateInput(input)

        if not self._should_run(input_coord):
            return self._create_result(input_coord.input)

        # extract the coordinates using the input
        lons, lats = self._extract_coordinates(input_coord)

        # add the extracted coordinates to the result
        return self._create_coordinate_result(input_coord, lons, lats)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        return {}, {}

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
        lats = input.input.get_data("lats", [])
        lons = input.input.get_data("lons", [])
        num_keypoints = min(len(lons), len(lats))
        print(f"num keypoints: {num_keypoints}")
        return num_keypoints < 2

    def _in_polygon(
        self, point: Tuple[float, float], polygon: List[Tuple[float, float]]
    ) -> bool:
        path = mpltPath.Path(polygon)
        return path.contains_point(point)


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
        lon_minmax = input.input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.input.get_request_info("lat_minmax", [0, 180])

        lon_pts, lat_pts = self._extract_lonlat(
            input, ocr_blocks, lon_minmax, lat_minmax
        )
        num_keypoints = min(len(lon_pts), len(lat_pts))
        if num_keypoints > 0:
            print(f"filtering via roi")
            # ----- do Region-of-Interest analysis (automatic cropping)
            roi_xy = input.input.get_data("roi")
            self._add_param(input.input, str(uuid.uuid4()), "roi", {"bounds": roi_xy})
            lon_pts, lat_pts = self._validate_lonlat_extractions(
                input, lon_pts, lat_pts, input.input.image.size, roi_xy
            )

        return lon_pts, lat_pts

    def _extract_lonlat(
        self,
        input: CoordinateInput,
        ocr_text_blocks_raw: DocTextExtraction,
        lon_minmax: List[float],
        lat_minmax: List[float],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        ocr_text_blocks = deepcopy(ocr_text_blocks_raw)
        for e in ocr_text_blocks.extractions:
            e.bounds = get_bounds_bounding_box(e.bounds)

        print("starting extraction")
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
                    # if block['text'] == '177⁹':
                    # print(f'groups: {m_groups}')
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
        print(
            f"new geo fence: {updated_geofence}\tlon clue: {lon_clue}\tlat clue: {lat_clue}"
        )

        # ---- finalize lat/lon results
        coord_results = []
        coord_lat_results = {}
        coord_lon_results = {}
        # ---- Check degress-minutes-seconds extractions...
        for idx, (groups, span, _) in dms_matches.items():
            # print(f'DMS MATCHES: {ocr_text_blocks[idx]["text"]}')
            try:
                deg = RE_NONNUMERIC.sub("", groups[0])
                deg = float(deg)
            except:
                continue  # not valid

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
            deg_decimal = deg + minutes / 60.0 + seconds / 3600.0

            # minutes and seconds expected in certain increments
            if not self._check_degree_increments(int(minutes), int(seconds)):
                print(
                    f"Excluding candidate point due to unexpected degree increment: {deg_decimal}"
                )
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    "coordinate-excluded",
                    {
                        "bounds": ocr_to_coordinates(
                            ocr_text_blocks.extractions[idx].bounds
                        ),
                        "text": ocr_text_blocks.extractions[idx].text,
                    },
                    "excluded due to invalid increment",
                )
                continue

            if self._check_consecutive(deg, minutes, seconds):
                print("Excluding candidate point: {}".format(deg_decimal))
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    "coordinate-excluded",
                    {
                        "bounds": ocr_to_coordinates(
                            ocr_text_blocks.extractions[idx].bounds
                        ),
                        "text": ocr_text_blocks.extractions[idx].text,
                    },
                    "excluded due to consecutive point",
                )
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
                        "lat keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print(
                        "Excluding candidate latitude point: {} with lat minmax {}".format(
                            deg_decimal, lat_minmax
                        )
                    )
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "latitude",
                        },
                        "excluded candidate lat point",
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
                        "lon keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print("Excluding candidate longitude point: {}".format(deg_decimal))
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "longitude",
                        },
                        "excluded candidate lon point",
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
                print("Excluding candidate point: {}".format(deg_decimal))
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    "coordinate-excluded",
                    {
                        "bounds": ocr_to_coordinates(
                            ocr_text_blocks.extractions[idx].bounds
                        ),
                        "text": ocr_text_blocks.extractions[idx].text,
                    },
                    "excluded due to consecutive point",
                )
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
                        "lat keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print("Excluding candidate latitude point: {}".format(deg_decimal))
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "latitude",
                        },
                        "excluded candidate lat point",
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
                        "lon keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print("Excluding candidate longitude point: {}".format(deg_decimal))
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "longitude",
                        },
                        "excluded candidate lon point",
                    )

        # ideally, we should have 3 or more keypoints for latitude and longitude each
        # to estimate map rotation, scale and translation tranformations
        if len(coord_lat_results) >= 3 and len(coord_lon_results) >= 3:
            for c in coord_results:
                self._add_param(
                    input.input,
                    str(uuid.uuid4()),
                    f"coordinate-{c.get_type()}",
                    {
                        "bounds": ocr_to_coordinates(c.get_bounds()),
                        "text": c.get_text(),
                        "parsed": c.get_parsed_degree(),
                        "type": "latitude" if c.is_lat() else "longitude",
                        "pixel_alignment": c.get_pixel_alignment(),
                    },
                    "extracted coordinate",
                )
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
                        "lat keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print("Excluding candidate latitude point: {}".format(deg_decimal))
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "latitude",
                        },
                        "excluded candidate lat point",
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
                        "lon keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        deg_decimal,
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
                    print("Excluding candidate longitude point: {}".format(deg_decimal))
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(
                                ocr_text_blocks.extractions[idx].bounds
                            ),
                            "text": ocr_text_blocks.extractions[idx].text,
                            "type": "longitude",
                        },
                        "excluded candidate lon point",
                    )

        for c in coord_results:
            self._add_param(
                input.input,
                str(uuid.uuid4()),
                f"coordinate-{c.get_type()}",
                {
                    "bounds": ocr_to_coordinates(c.get_bounds()),
                    "text": c.get_text(),
                    "parsed": c.get_parsed_degree(),
                    "type": "latitude" if c.is_lat() else "longitude",
                    "pixel_alignment": c.get_pixel_alignment(),
                },
                "extracted coordinate",
            )

        return (coord_lon_results, coord_lat_results)

    def _check_degree_increments(self, minutes: int, seconds: int) -> bool:
        # support 1/16 or 1/12 degree increments
        return (minutes * 60 + seconds) % 225 == 0 or (
            minutes * 60 + seconds
        ) % 300 == 0

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

    def _validate_lonlat_extractions(
        self,
        input: CoordinateInput,
        lon_results: Dict[Tuple[float, float], Coordinate],
        lat_results: Dict[Tuple[float, float], Coordinate],
        im_size: Tuple[float, float],
        roi_xy: List[Tuple[float, float]] = [],
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        print("validating lonlat")

        num_lat_pts = len(lat_results)
        num_lon_pts = len(lon_results)

        # remove points in secondary maps
        # lon_results, _ = self._filter_secondary_map(lon_results)
        # lat_results, _ = self._filter_secondary_map(lat_results)
        # num_lat_pts = len(lat_results)
        # num_lon_pts = len(lon_results)

        if roi_xy and (num_lat_pts > 4 or num_lon_pts > 4):
            for (deg, y), coord in list(lat_results.items()):
                if not self._in_polygon(coord.get_pixel_alignment(), roi_xy):
                    print(
                        f"Excluding out-of-bounds latitude point: {deg} ({coord.get_pixel_alignment()})"
                    )
                    del lat_results[(deg, y)]
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(coord.get_bounds()),
                            "text": coord.get_text(),
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "excluded due to being outside roi",
                    )
            for (deg, x), coord in list(lon_results.items()):
                if not self._in_polygon(coord.get_pixel_alignment(), roi_xy):
                    print(
                        f"Excluding out-of-bounds longitude point: {deg} ({coord.get_pixel_alignment()})"
                    )
                    del lon_results[(deg, x)]
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(coord.get_bounds()),
                            "text": coord.get_text(),
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "excluded due to being outside roi",
                    )

        num_lat_pts = len(lat_results)
        num_lon_pts = len(lon_results)
        print(f"after exclusion lat,lon: {num_lat_pts},{num_lon_pts}")

        # if num_lon_pts > 4:
        #    lon_results = self._remove_outlier_pts(input, lon_results, im_size[0], im_size[1])
        # if num_lat_pts > 4:
        #    lat_results = self._remove_outlier_pts(input, lat_results, im_size[1], im_size[0])

        # check number of unique lat and lon values
        num_lat_pts = len(set([x[0] for x in lat_results]))
        num_lon_pts = len(set([x[0] for x in lon_results]))
        print(f"distinct after outlier lat,lon: {num_lat_pts},{num_lon_pts}")

        if num_lon_pts >= 2 and num_lat_pts == 1:
            # estimate additional lat pt (based on lon pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[1]) for k, v in lon_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lat_pt = list(lat_results.items())[0]
                # new_y = im_size[1]-1
                new_y = 0 if lat_pt[0][1] > im_size[1] / 2 else im_size[1] - 1
                print("t1")
                new_lat = -deg_per_pxl * (new_y - lat_pt[0][1]) + lat_pt[0][0]
                coord = Coordinate(
                    "lat keypoint",
                    "",
                    new_lat,
                    True,
                    pixel_alignment=(lat_pt[1].to_deg_result()[1], new_y),
                    confidence=0.6,
                )
                print("t2")
                lat_results[(new_lat, new_y)] = coord

        elif num_lat_pts >= 2 and num_lon_pts == 1:
            # estimate additional lon pt (based on lat pxl resolution)
            lst = [
                (k[0], k[1], v.get_pixel_alignment()[0]) for k, v in lat_results.items()
            ]
            max_pt = max(lst, key=lambda p: p[1])
            min_pt = min(lst, key=lambda p: p[1])
            pxl_range = max_pt[1] - min_pt[1]
            deg_range = max_pt[0] - min_pt[0]
            if deg_range != 0 and pxl_range != 0:
                deg_per_pxl = abs(
                    deg_range / pxl_range
                )  # TODO could use geodesic dist here?
                lon_pt = list(lon_results.items())[0]
                # new_x = im_size[0]-1
                new_x = 0 if lon_pt[0][1] > im_size[0] / 2 else im_size[0] - 1
                print("t3")
                new_lon = -deg_per_pxl * (new_x - lon_pt[0][1]) + lon_pt[0][0]
                coord = Coordinate(
                    "lon keypoint",
                    "",
                    new_lon,
                    False,
                    pixel_alignment=(new_x, lon_pt[1].to_deg_result()[1]),
                    confidence=0.6,
                )
                print(f"lon pt: {lon_pt}")
                print("t4")
                lon_results[(new_lon, new_x)] = coord
        print("done validating")

        return (lon_results, lat_results)

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
                else:
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        "coordinate-excluded",
                        {
                            "bounds": ocr_to_coordinates(v.get_bounds()),
                            "text": v.get_text(),
                            "type": "latitude" if v.is_lat() else "longitude",
                            "pixel_alignment": v.get_pixel_alignment(),
                        },
                        "excluded due to being an outlier",
                    )
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
            print(f"min coord: {min_coord}\tmax coord: {max_coord}")
            print(
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
                    print(f"dropping {r} due to being part of secondary map")
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


class GeocodeCoordinatesExtractor(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        roi_xy = input.input.get_data("roi")
        ocr_blocks = input.input.get_data("ocr_blocks")
        lon_minmax = input.input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.input.get_request_info("lat_minmax", [0, 90])
        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")
        lon_sign_factor = input.input.get_request_info("lon_sign_factor", 1)

        # not enough key-points, try geo-coding
        lon_pts, lat_pts = self._extract_places(
            ocr_blocks,
            (lon_pts, lat_pts),
            lon_minmax,
            lat_minmax,
            lon_sign_factor,
            roi_xy,
            use_google_maps=True,
            geocode_path=os.path.join(GEOCODE_CACHE, input.input.raster_id + ".pkl"),
        )

        return lon_pts, lat_pts

    def _extract_places(
        self,
        ocr_text_blocks: DocTextExtraction,
        lonlat_results: Tuple[
            Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
        ],
        lon_minmax: List[float],
        lat_minmax: List[float],
        lon_sign_factor: float = 1.0,
        roi_xy: List[Tuple[float, float]] = [],
        use_google_maps: bool = True,
        geocode_path: str = "",
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        if not ocr_text_blocks or len(ocr_text_blocks.extractions) == 0:
            print("WARNING! No ocr text blocks available!")
            return ({}, {})

        google_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", None)
        using_google = use_google_maps and google_api_key

        lon_clue = (
            lon_minmax[0] + lon_minmax[1]
        ) / 2  # mid-points of lon/lat hint area
        lat_clue = (lat_minmax[0] + lat_minmax[1]) / 2

        lon_range = (
            abs(lon_minmax[1] - lon_minmax[0]) / 2
        ) * 0.5  # use a smaller fov for geo-coding!
        lat_range = (abs(lat_minmax[1] - lat_minmax[0]) / 2) * 0.5
        lon_minmax = [lon_clue - lon_range, lon_clue + lon_range]
        lat_minmax = [lat_clue - lat_range, lat_clue + lat_range]

        geocoder = None
        timeout = 10
        if using_google:
            geocoder = GoogleV3(api_key=google_api_key, domain="maps.googleapis.com")
        else:
            geocoder = Nominatim(
                timeout=timeout, user_agent="uncharted-cma-challenge-geocoder"  # type: ignore
            )

        geocode_cache = {}
        if geocode_path:
            geocode_cache = load_geocode_cache(geocode_path)

        lon_results, lat_results = lonlat_results
        geo_lonlat_results = []
        for idx, block in enumerate(ocr_text_blocks.extractions):
            if len(block.text) <= MIN_PLACENAME_LEN:
                continue
            re_result = RE_LETTERS.search(block.text)
            if not re_result:
                continue

            if roi_xy:
                x = self._get_center_x(block.bounds, (0.0, 1.0))
                y = self._get_center_y(block.bounds)
                if not self._in_bounds(x, roi_xy[0]) or not self._in_bounds(
                    y, roi_xy[1]
                ):
                    # this OCR block is not in ROI, so skip geocoding for this OCR block
                    continue

            viewbox = [
                Point(lat_minmax[1], lon_sign_factor * lon_minmax[1]),
                Point(lat_minmax[0], lon_sign_factor * lon_minmax[0]),
            ]

            places = geocode_cache.get(block.text, None)
            if places is None:
                if using_google:
                    # run Google geo-coder
                    places = geocoder.geocode(
                        block.text,  # type: ignore
                        exactly_one=False,  # type: ignore
                        timeout=timeout,  # type: ignore
                        region="us",  # type: ignore
                        bounds=viewbox,  # type: ignore
                    )
                else:
                    # run Nominatim geo-coder
                    # featuretype='settlement' #'city,settlement'
                    places = geocoder.geocode(
                        block.text,  # type: ignore
                        exactly_one=False,  # type: ignore
                        limit=4,  # type: ignore
                        country_codes="us",  # type: ignore
                        viewbox=viewbox,  # type: ignore
                    )  # featuretype=featuretype)
                if places:
                    geocode_cache[block.text].extend(places)  # type: ignore
                else:
                    geocode_cache[block.text].append(None)

            if places is None:
                continue

            place_dists = []
            for p_idx, place in enumerate(places):  # type: ignore
                if place is None:
                    continue

                this_lon = place.longitude * lon_sign_factor
                this_lat = place.latitude
                if (
                    this_lon >= lon_minmax[0]
                    and this_lon <= lon_minmax[1]
                    and this_lat >= lat_minmax[0]
                    and this_lat <= lat_minmax[1]
                ):
                    # got a geo-coding result within target geo-fence...
                    # check if location 'type' is ok (google geocoding API only)
                    place_type_ok = False
                    if using_google:
                        place_types = []
                        try:
                            place_types = place.raw["types"]
                        except:
                            place_types = []

                        for place_type in place_types:
                            if place_type in LOC_TYPES_PASSLIST_GOOGLE:
                                place_type_ok = True
                    else:
                        place_type = ""
                        try:
                            loc_class = place.raw.get("class", "")
                            if loc_class == "place":
                                place_type = place.raw.get("type", "")
                        except:
                            place_type = ""
                        if place_type in LOC_TYPES_PASSLIST_NOM:
                            place_type_ok = True

                    if place_type_ok:
                        # save keypoint
                        xd = this_lon - lon_clue
                        yd = this_lat - lat_clue
                        place_dists.append((p_idx, xd * xd + yd * yd))

            if place_dists:
                # choose match closest to 'clue' lat/lon pt
                p_idx_match, p_dist_sq = min(place_dists, key=lambda x: x[1])
                x_pixel = self._get_center_x(
                    ocr_text_blocks.extractions[idx].bounds, (0.0, 1.0)
                )
                y_pixel = self._get_center_y(ocr_text_blocks.extractions[idx].bounds)
                lonlat_deg = (
                    places[p_idx_match].longitude * lon_sign_factor,  # type: ignore
                    places[p_idx_match].latitude,  # type: ignore
                )

                geo_lonlat_results.append((lonlat_deg, (x_pixel, y_pixel), p_dist_sq))

            if len(geo_lonlat_results) >= 5:
                # we have sufficient keypoint results
                break

        if geocode_path:
            cache_geocode_data(geocode_cache, geocode_path)

        # sort geocode results by distance from clue pt
        geo_lonlat_results.sort(key=lambda x: x[2])

        if len(geo_lonlat_results) == 0:
            return lon_results, lat_results

        # goal is to retain up to 3 keypoints
        geo_pts_needed = max(min(3 - len(lon_results), len(geo_lonlat_results)), 0)
        if geo_pts_needed > 0:
            for i in range(geo_pts_needed):
                lon_deg = geo_lonlat_results[i][0][0]
                x = geo_lonlat_results[i][1][0]
                y = geo_lonlat_results[i][1][1]
                coord = Coordinate(
                    "lon keypoint",
                    "",
                    lon_deg,
                    False,
                    pixel_alignment=(x, y),
                    confidence=0.55,
                )
                lon_results[(lon_deg, x)] = coord

        geo_pts_needed = max(min(3 - len(lat_results), len(geo_lonlat_results)), 0)
        if geo_pts_needed > 0:
            for i in range(geo_pts_needed):
                lat_deg = geo_lonlat_results[i][0][1]
                x = geo_lonlat_results[i][1][0]
                y = geo_lonlat_results[i][1][1]
                coord = Coordinate(
                    "lat keypoint",
                    "",
                    lat_deg,
                    True,
                    pixel_alignment=(x, y),
                    confidence=0.55,
                )
                lat_results[(lat_deg, y)] = coord

        return lon_results, lat_results

    def _get_center_y(self, bounding_poly: List[TPoint]) -> float:
        min_y = bounding_poly[0].y
        max_y = bounding_poly[3].y
        return (min_y + max_y) / 2.0

    def _get_center_x(
        self, bounding_poly: List[TPoint], x_ranges: Tuple[float, float]
    ) -> float:
        min_x = bounding_poly[0].x
        max_x = bounding_poly[2].x
        if x_ranges[0] > 0.0 or x_ranges[1] < 1.0:
            x_span = max_x - min_x
            min_x += x_span * x_ranges[0]
            max_x -= x_span * (1.0 - x_ranges[1])
        return (min_x + max_x) / 2.0

    def _in_bounds(self, value: float, bounds: Tuple[float, float]) -> bool:
        return value >= bounds[0] and value <= bounds[1]
