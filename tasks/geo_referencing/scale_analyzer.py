import logging
from geopy.distance import distance as geo_distance
from tasks.common.task import Task, TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.geo_referencing.entities import (
    DocGeoFence,
    MapScale,
    GEOFENCE_OUTPUT_KEY,
    MAP_SCALE_OUTPUT_KEY,
)

from tasks.common.task import Task, TaskInput, TaskResult
from typing import Tuple

# assumed scan resolution of the input raster (Note: all NGMDB products are scanned at 300 DPI)
DPI_DEFAULT = 300
# min and max valid ranges for printed map scale (eg "1:VALUE")
PIXEL_SCALE_MIN = 10000
PIXEL_SCALE_MAX = 2000000

KM_PER_INCH = 2.54e-5


logger = logging.getLogger("scale_analyzer")


class ScaleAnalyzer(Task):
    """
    Analyze and extract the expected pixel-to-km scale for the map
    using info from the metadata and geofence tasks
    """

    def __init__(self, task_id: str, dpi: int = DPI_DEFAULT):
        self._dpi = dpi
        super().__init__(task_id)

    def run(self, input: TaskInput) -> TaskResult:

        # get metadata result
        metadata: MetadataExtraction = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        # get geofence result
        geofence: DocGeoFence = input.parse_data(
            GEOFENCE_OUTPUT_KEY, DocGeoFence.model_validate
        )

        scale_raw = ""
        scale_pixels = 0.0
        km_per_pixel = 0.0
        deg_per_pixel = 0.0
        if metadata:
            # normalize the extracted scale
            scale_raw = metadata.scale.lower().strip()
            scale_pixels = self._normalize_scale(scale_raw)
            if scale_pixels > 0:
                logger.info(f"Map pixel scale extracted as: 1 to {int(scale_pixels)}")

            # convert scale to the expected km / pixel
            if self._dpi > 0.0:
                km_per_pixel = scale_pixels * KM_PER_INCH / self._dpi

                lon_c = 0.0
                lat_c = 0.0
                if geofence:
                    lon_c = (
                        geofence.geofence.lon_minmax[0]
                        + geofence.geofence.lon_minmax[1]
                    ) / 2
                    lat_c = (
                        geofence.geofence.lat_minmax[0]
                        + geofence.geofence.lat_minmax[1]
                    ) / 2
                # calc expected degrees per pixel
                deg_per_pixel = self._calc_deg_per_pixel((lon_c, lat_c), km_per_pixel)

        # generate the map scale result
        scale_result = MapScale(
            scale_raw=scale_raw,
            dpi=self._dpi,
            scale_pixels=scale_pixels,
            km_per_pixel=km_per_pixel,
            degrees_per_pixel=deg_per_pixel,
        )
        result = self._create_result(input)
        result.add_output(MAP_SCALE_OUTPUT_KEY, scale_result.model_dump())
        return result

    def _normalize_scale(self, scale_raw: str) -> float:
        """
        extract and normalize the raw scale string
        (assumed format is "1:VALUE"
        """
        scale = 0.0
        try:
            if not scale_raw.startswith("1:"):
                return scale
            scale = float(scale_raw[2:])
            if scale < PIXEL_SCALE_MIN or scale > PIXEL_SCALE_MAX:
                logger.warning(
                    f"Extracted scale is: {scale_raw}; which is outside the valid range so disregarding"
                )
                return 0.0
        except Exception as e:
            logger.warning(
                f"Unable to parse a valid scale due to: {repr(e)}; disregarding"
            )
            return 0.0
        return scale

    def _calc_deg_per_pixel(
        self, lonlat_pt: Tuple[float, float], km_per_pixel: float
    ) -> float:
        """
        Estimate the degrees-per-pixel resolution for a map given a lon/lat point of interest
        """
        try:
            pt_north = geo_distance(kilometers=1).destination(
                (lonlat_pt[1], lonlat_pt[0]), bearing=0
            )
            pt_east = geo_distance(kilometers=1).destination(
                (lonlat_pt[1], lonlat_pt[0]), bearing=90
            )
            # degrees per km is avg of lat and lon differences at the pt of interest
            deg_per_km = (
                abs(pt_north.latitude - lonlat_pt[1])
                + abs(pt_east.longitude - lonlat_pt[0])
            ) / 2

            return km_per_pixel * deg_per_km
        except Exception as e:
            logger.warning(f"Exception calculating degrees-per-pixel: {repr(e)}")
            return 0
