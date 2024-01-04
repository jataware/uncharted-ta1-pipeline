import re
import utm
import uuid

from copy import deepcopy

from tasks.geo_referencing.coordinates_extractor import (
    CoordinatesExtractor,
    CoordinateInput,
)
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.geo_referencing.entities import Coordinate
from tasks.geo_referencing.util import ocr_to_coordinates, get_bounds_bounding_box

from typing import Any, Dict, List, Tuple

# UTM coordinates
# pre-compiled regex patterns
RE_NORTHEAST = re.compile(
    r"^|\b([1-9]?[ ,.]?[0-9]{3}[ ,.]?[0-9]00) ?m?(N|North|E|East)?($|\b)"
)

RE_NONNUMERIC = re.compile(r"[^0-9]")

FOV_RANGE_METERS = 200000  # fallback search range (meters)


class UTMCoordinatesExtractor(CoordinatesExtractor):
    def __init__(self, task_id: str):
        super().__init__(task_id)

    def _extract_coordinates(
        self, input: CoordinateInput
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        ocr_blocks = input.input.get_data("ocr_blocks")
        lon_minmax = input.input.get_request_info("lon_minmax", [0, 180])
        lat_minmax = input.input.get_request_info("lat_minmax", [-80, 84])
        lon_pts = input.input.get_data("lons")
        lat_pts = input.input.get_data("lats")
        lon_sign_factor = input.input.get_request_info("lon_sign_factor", 1)

        # UTM is limited to the range of 80 south to 84 north
        if lat_minmax[0] < -80:
            lat_minmax[0] = -80
        if lat_minmax[1] > 84:
            lat_minmax[1] = 84
        print(f"utm lon & lat limits: {lon_minmax}\t{lat_minmax}")

        lon_pts, lat_pts = self._extract_utm(
            input,
            ocr_blocks,
            (lon_pts, lat_pts),
            lon_minmax,
            lat_minmax,
            lon_sign_factor,
        )

        return lon_pts, lat_pts

    def _extract_utm(
        self,
        input: CoordinateInput,
        ocr_text_blocks_raw: DocTextExtraction,
        lonlat_results: Tuple[
            Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
        ],
        lon_minmax: List[float],
        lat_minmax: List[float],
        lon_sign_factor: float = 1.0,
    ) -> Tuple[
        Dict[Tuple[float, float], Coordinate], Dict[Tuple[float, float], Coordinate]
    ]:
        if not ocr_text_blocks_raw or len(ocr_text_blocks_raw.extractions) == 0:
            print("WARNING! No ocr text blocks available!")
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
        print(f"lat clue: {lat_clue}\tlon clue: {lon_clue}")
        utm_clue = utm.from_latlon(lat_clue, lon_clue * lon_sign_factor)
        easting_clue = utm_clue[0]
        northing_clue = utm_clue[1]
        northing_range0 = utm.from_latlon(lat_minmax[0], lon_clue * lon_sign_factor)
        northing_range = min(
            abs(utm_clue[1] - northing_range0[1]), FOV_RANGE_METERS
        )  # northing search range (in meters)
        easting_range0 = utm.from_latlon(lat_clue, lon_minmax[0] * lon_sign_factor)
        easting_range = min(
            abs(utm_clue[0] - easting_range0[0]), FOV_RANGE_METERS
        )  # easting search range (in meters)

        utm_geofence_min = utm.from_latlon(lat_minmax[0], lon_minmax[0])
        utm_geofence_max = utm.from_latlon(lat_minmax[1], lon_minmax[1])

        idx = 0
        ne_matches: List[Tuple[int, Any, Tuple[int, int, int]]] = []
        for block in ocr_text_blocks.extractions:
            matches_iter = RE_NORTHEAST.finditer(block.text)
            for m in matches_iter:
                m_groups = m.groups()
                if any(x for x in m_groups):
                    # valid match
                    m_span = (m.start(), m.end(), len(block.text))
                    ne_matches.append((idx, m_groups, m_span))
            idx += 1

        # ---- Check Northing-Easting extractions...
        for idx, groups, span in ne_matches:
            utm_dist = RE_NONNUMERIC.sub("", groups[0])
            utm_dist = float(utm_dist)
            if utm_dist == 0:  # skip noisy extraction
                continue

            if abs(utm_dist - northing_clue) < abs(utm_dist - easting_clue):
                # latitude keypoint (y-axis)
                if (
                    utm_dist >= northing_clue - northing_range
                    and utm_dist <= northing_clue + northing_range
                ):
                    # valid latitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    # convert extracted northing value to latitude and save keypoint result
                    latlon_pt = utm.to_latlon(
                        easting_clue, utm_dist, utm_clue[2], utm_clue[3]
                    )
                    coord = Coordinate(
                        "lat keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        latlon_pt[0],
                        True,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.75,
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
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "extracted northing utm coordinate",
                    )
                elif utm_geofence_min[1] <= utm_dist <= utm_geofence_max[1]:
                    # valid latitude point
                    # TODO: LOWER CONFIDENCE SCORE MATCH
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    # convert extracted northing value to latitude and save keypoint result
                    latlon_pt = utm.to_latlon(
                        easting_clue, utm_dist, utm_clue[2], utm_clue[3]
                    )
                    coord = Coordinate(
                        "lat keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        latlon_pt[0],
                        True,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.75,
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
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "extracted northing utm coordinate",
                    )
                else:
                    print("Excluding candidate northing point: {}".format(utm_dist))
            else:
                # longitude keypoint (x-axis)
                if (
                    utm_dist >= easting_clue - easting_range
                    and utm_dist <= easting_clue + easting_range
                ):
                    # valid longitude point
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    # convert extracted easting value to longitude and save keypoint result
                    latlon_pt = utm.to_latlon(
                        utm_dist, northing_clue, utm_clue[2], utm_clue[3]
                    )
                    coord = Coordinate(
                        "lon keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        latlon_pt[1] * lon_sign_factor,
                        False,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.75,
                    )
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    lon_results[(latlon_pt[1] * lon_sign_factor, x_pixel)] = coord
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        f"coordinate-{coord.get_type()}",
                        {
                            "bounds": ocr_to_coordinates(coord.get_bounds()),
                            "text": coord.get_text(),
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "extracted easting utm coordinate",
                    )
                elif utm_geofence_min[0] <= utm_dist <= utm_geofence_max[0]:
                    # valid longitude point
                    # TODO: LOWER CONFIDENCE SCORE MATCH
                    x_ranges = (
                        (0.0, 1.0)
                        if span[2] == 0
                        else (span[0] / float(span[2]), span[1] / float(span[2]))
                    )
                    # convert extracted easting value to longitude and save keypoint result
                    latlon_pt = utm.to_latlon(
                        utm_dist, northing_clue, utm_clue[2], utm_clue[3]
                    )
                    coord = Coordinate(
                        "lon keypoint",
                        ocr_text_blocks.extractions[idx].text,
                        latlon_pt[1] * lon_sign_factor,
                        False,
                        ocr_text_blocks.extractions[idx].bounds,
                        x_ranges=x_ranges,
                        confidence=0.75,
                    )
                    x_pixel, y_pixel = coord.get_pixel_alignment()
                    lon_results[(latlon_pt[1] * lon_sign_factor, x_pixel)] = coord
                    self._add_param(
                        input.input,
                        str(uuid.uuid4()),
                        f"coordinate-{coord.get_type()}",
                        {
                            "bounds": ocr_to_coordinates(coord.get_bounds()),
                            "text": coord.get_text(),
                            "type": "latitude" if coord.is_lat() else "longitude",
                            "pixel_alignment": coord.get_pixel_alignment(),
                        },
                        "extracted easting utm coordinate",
                    )
                else:
                    print("Excluding candidate easting point: {}".format(utm_dist))
        print("done utm")

        return (lon_results, lat_results)
