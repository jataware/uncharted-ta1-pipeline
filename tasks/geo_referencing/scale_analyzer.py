import logging
from geopy.distance import distance as geo_distance
from PIL.Image import Image
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

    def __init__(self, task_id: str, dpi_default: int = DPI_DEFAULT):
        self._dpi_default = dpi_default
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

        # get input raster DPI
        dpi = self._get_raster_dpi(input.image)

        scale_raw = ""
        scale_pixels = 0.0
        km_per_pixel = 0.0
        lon_per_km = 0.0
        lat_per_km = 0.0
        if metadata:
            # normalize the extracted scale
            scale_raw = metadata.scale.lower().strip()
            scale_pixels = self._normalize_scale(scale_raw)
            if scale_pixels > 0:
                logger.info(f"Map pixel scale extracted as: 1 to {int(scale_pixels)}")

            # convert scale to the expected km / pixel
            if dpi > 0.0:
                km_per_pixel = scale_pixels * KM_PER_INCH / dpi

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
                lon_per_km, lat_per_km = ScaleAnalyzer.calc_deg_per_km((lon_c, lat_c))

        # generate the map scale result
        scale_result = MapScale(
            scale_raw=scale_raw,
            dpi=dpi,
            scale_pixels=scale_pixels,
            km_per_pixel=km_per_pixel,
            lonlat_per_pixel=(lon_per_km * km_per_pixel, lat_per_km * km_per_pixel),
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
            # extract scale denominator, remove puncuation and convert to float
            scale_str = scale_raw[2:].replace(" ", "").replace(",", "")
            scale = float(scale_str)
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

    def _get_raster_dpi(self, im: Image) -> int:
        """
        Parse the input raster's DPI (dots-per-inch) value
        """

        DPI_MIN = 72
        try:
            dpi_xy: Tuple[int, int] = im.info.get("dpi", [0, 0])
            # use the average x,y DPI value, if they are slightly different
            dpi = int((dpi_xy[0] + dpi_xy[1]) / 2)
            if dpi < DPI_MIN:
                logger.info(
                    f"Input raster DPI is {dpi} which is too low, using default DPI value for scale analysis: {self._dpi_default}"
                )
                dpi = self._dpi_default
            else:
                logger.info(f"Input raster DPI: {dpi}")
            return dpi
        except Exception as ex:
            logger.warning(
                f"Exception parsing raster DPI; using default DPI value for scale analysis: {self._dpi_default} - {repr(ex)}"
            )
            return self._dpi_default

    @staticmethod
    def calc_deg_per_km(lonlat_pt: Tuple[float, float]) -> Tuple[float, float]:
        """
        Estimate the degrees-per-km resolution for a lon/lat point of interest
        Returns a tuple: (lon_per_km, lat_per_km)
        """
        try:
            pt_north = geo_distance(kilometers=1).destination(
                (lonlat_pt[1], lonlat_pt[0]), bearing=0
            )
            pt_east = geo_distance(kilometers=1).destination(
                (lonlat_pt[1], lonlat_pt[0]), bearing=90
            )

            lon_per_km = abs(pt_east.longitude - lonlat_pt[0])
            lat_per_km = abs(pt_north.latitude - lonlat_pt[1])

            return (lon_per_km, lat_per_km)
        except Exception as e:
            logger.warning(f"Exception calculating degrees-per-km: {repr(e)}")
            return (0.0, 0.0)
