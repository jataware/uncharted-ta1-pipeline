import os
import logging

from pathlib import Path

from pipelines.geo_referencing.output import (
    OutputCreator,
    GCPOutput,
    GeoReferencingOutput,
    ProjectedMapOutput,
    UserLeverOutput,
    SummaryOutput,
)
from tasks.common.pipeline import Pipeline
from tasks.common.task import TaskInput
from tasks.geo_referencing.coordinates_extractor import (
    GeoCoordinatesExtractor,
)
from tasks.geo_referencing.corner_point_extractor import CornerPointExtractor
from tasks.geo_referencing.entities import PROJECTED_MAP_OUTPUT_KEY
from tasks.geo_referencing.state_plane_extractor import StatePlaneExtractor
from tasks.geo_referencing.utm_extractor import UTMCoordinatesExtractor
from tasks.geo_referencing.filter import (
    NaiveFilter,
    OutlierFilter,
    ROIFilter,
    UTMStatePlaneFilter,
    DistinctDegreeOutlierFilter,
    HighQualityCoordinateFilter,
)
from tasks.geo_referencing.geo_fencing import GeoFencer
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.geocode import PointGeocoder, BoxGeocoder
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
from tasks.geo_referencing.inference import InferenceCoordinateExtractor
from tasks.geo_referencing.roi_extractor import (
    ModelROIExtractor,
    buffer_fixed,
)
from tasks.metadata_extraction.geocoder import Geocoder, NominatimGeocoder
from tasks.metadata_extraction.metadata_extraction import MetadataExtractor, LLM
from tasks.metadata_extraction.scale import ScaleExtractor
from tasks.metadata_extraction.text_filter import (
    FilterMode,
    TextFilter,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor

from typing import List

logger = logging.getLogger("factory")


def run_step(input: TaskInput) -> bool:
    lats = input.get_data("lats", {})
    lons = input.get_data("lons", {})

    lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lats.items()))
    lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lons.items()))
    num_keypoints = min(len(lons_distinct), len(lats_distinct))
    logger.info(f"running step due to insufficient key points: {num_keypoints < 2}")
    return num_keypoints < 2


def create_geo_referencing_pipelines(
    working_dir: str,
    segmentation_model_path: str,
    state_plane_lookup_filename: str,
    state_plane_zone_filename: str,
    state_code_filename: str,
    country_code_filename: str,
    ocr_gamma_correction: float,
    gpu_enabled: bool,
) -> List[Pipeline]:
    geocoding_cache_bounds = os.path.join(working_dir, "geocoding_cache_bounds.json")
    geocoding_cache_points = os.path.join(working_dir, "geocoding_cache_points.json")
    geocoder_bounds = NominatimGeocoder(
        10, geocoding_cache_bounds, 1, country_code_filename=country_code_filename
    )
    geocoder_points = NominatimGeocoder(
        10, geocoding_cache_points, 5, country_code_filename=country_code_filename
    )

    segmentation_cache = os.path.join(working_dir, "segmentation")
    text_cache = os.path.join(working_dir, "text")
    metadata_cache = os.path.join(working_dir, f"metadata-gamma-{ocr_gamma_correction}")
    geocoder_thresh = 10

    p = []

    tasks = []
    tasks.append(
        TileTextExtractor(
            "first", Path(text_cache), 6000, gamma_correction=ocr_gamma_correction
        )
    )
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            segmentation_cache,
            confidence_thres=0.25,
            gpu=gpu_enabled,
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_fixed,
        )
    )
    tasks.append(
        TextFilter(
            "text_filter",
            # input_key="metadata_ocr",
            output_key="filtered_ocr_text",
            classes=[
                "cross_section",
                "legend_points_lines",
                "legend_polygons",
            ],
        )
    )
    tasks.append(
        MetadataExtractor(
            "metadata_extractor",
            LLM.GPT_4_O,
            "filtered_ocr_text",
            cache_dir=metadata_cache,
        )
    )
    tasks.append(
        Geocoder(
            "geo-bounds",
            geocoder_bounds,
            run_bounds=True,
            run_points=False,
            run_centres=False,
        )
    )
    tasks.append(GeoFencer("geofence"))
    tasks.append(GeoCoordinatesExtractor("geo_coordinates_extractor"))
    tasks.append(ROIFilter("roi_filter"))
    tasks.append(DistinctDegreeOutlierFilter("uniqueness_filter"))
    tasks.append(HighQualityCoordinateFilter("quality_filter"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    tasks.append(
        TextFilter(
            "map_area_filter",
            FilterMode.INCLUDE,
            TEXT_EXTRACTION_OUTPUT_KEY,
            "map_area_filter",
            ["map"],
            run_step,
        )
    )
    tasks.append(
        MetadataExtractor(
            "metadata_map_area_extractor",
            LLM.GPT_4_O,
            "map_area_filter",
            run_step,
            include_place_bounds=True,
        )
    )
    tasks.append(
        Geocoder(
            "geo-places",
            geocoder_points,
            run_bounds=False,
            run_points=True,
            run_centres=False,
            should_run=run_step,
        )
    )
    tasks.append(
        Geocoder(
            "geo-centres",
            geocoder_bounds,
            run_bounds=False,
            run_points=False,
            run_centres=True,
            should_run=run_step,
        )
    )
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(
        StatePlaneExtractor(
            "great-plains",
            state_plane_lookup_filename,
            state_plane_zone_filename,
            state_code_filename,
        )
    )
    tasks.append(OutlierFilter("utm-outliers"))
    tasks.append(UTMStatePlaneFilter("utm-state-plane"))
    tasks.append(
        PointGeocoder(
            "geocoded-georeferencing", ["point", "population"], geocoder_thresh
        )
    )
    tasks.append(BoxGeocoder("geocoded-box", ["point", "population"], geocoder_thresh))
    tasks.append(CornerPointExtractor("corner_point_extractor"))
    tasks.append(InferenceCoordinateExtractor("coordinate-inference"))
    tasks.append(ScaleExtractor("scaler", ""))
    tasks.append(CreateGroundControlPoints("seventh", create_random_pts=False))
    tasks.append(GeoReference("eighth", 1))
    p.append(
        Pipeline(
            "roi poly fixed",
            "roi poly",
            [
                GeoReferencingOutput("geo"),
                SummaryOutput("summary"),
                UserLeverOutput("levers"),
                GCPOutput("gcps"),
                ProjectedMapOutput(PROJECTED_MAP_OUTPUT_KEY),
            ],
            tasks,
        )
    )

    return p


def create_geo_referencing_pipeline(
    segmentation_model_path: str,
    outputs: List[OutputCreator],
    working_dir: str,
    state_plane_lookup_filename: str,
    state_plane_zone_filename: str,
    state_code_filename: str,
    country_code_filename: str,
    ocr_gamma_correction: float,
    gpu_enabled: bool,
) -> Pipeline:
    geocoding_cache_bounds = os.path.join(working_dir, "geocoding_cache_bounds.json")
    geocoding_cache_points = os.path.join(working_dir, "geocoding_cache_points.json")
    geocoder_bounds = NominatimGeocoder(
        10, geocoding_cache_bounds, 1, country_code_filename=country_code_filename
    )
    geocoder_points = NominatimGeocoder(
        10, geocoding_cache_points, 5, country_code_filename=country_code_filename
    )
    segmentation_cache = os.path.join(working_dir, "segmentation")
    text_cache = os.path.join(working_dir, "text")
    metadata_cache = os.path.join(working_dir, "metadata")

    tasks = []
    tasks.append(
        ResizeTextExtractor(
            "resize_text_extractor",
            Path(text_cache),
            False,
            True,
            6000,
            gamma_correction=ocr_gamma_correction,
            output_key="metadata_ocr",
        )
    )
    tasks.append(
        TileTextExtractor(
            "tile_text_extractor",
            Path(text_cache),
            6000,
            gamma_correction=ocr_gamma_correction,
        )
    )
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            segmentation_cache,
            confidence_thres=0.25,
            gpu=gpu_enabled,
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model_roi_buffering",
            buffer_fixed,
        )
    )
    tasks.append(
        TextFilter(
            "resize_text_filter",
            input_key="metadata_ocr",
            output_key="filtered_ocr_text",
            classes=[
                "cross_section",
                "legend_points_lines",
                "legend_polygons",
            ],
        )
    )
    tasks.append(
        MetadataExtractor(
            "metadata_extractor",
            LLM.GPT_4_O,
            "filtered_ocr_text",
            cache_dir=metadata_cache,
        )
    )
    tasks.append(
        Geocoder(
            "geobounds",
            geocoder_bounds,
            run_bounds=True,
            run_points=False,
            run_centres=False,
        )
    )
    tasks.append(GeoFencer("geofence"))
    tasks.append(GeoCoordinatesExtractor("geocoodinates_extractor"))
    tasks.append(ROIFilter("roiness"))
    tasks.append(DistinctDegreeOutlierFilter("uniqueness"))
    tasks.append(OutlierFilter("coordinate_outlier_filter"))
    tasks.append(NaiveFilter("coordinate_naive_filter"))
    tasks.append(
        TextFilter(
            "tiled_map_area_filter",
            FilterMode.INCLUDE,
            TEXT_EXTRACTION_OUTPUT_KEY,
            "map_area_filter",
            ["map"],
            run_step,
        )
    )
    tasks.append(
        MetadataExtractor(
            "metadata_map_area_extractor",
            LLM.GPT_4_O,
            "map_area_filter",
            run_step,
            include_place_bounds=True,
        )
    )
    tasks.append(
        Geocoder(
            "places_geocoder",
            geocoder_points,
            run_bounds=False,
            run_points=True,
            run_centres=False,
        )
    )
    tasks.append(
        Geocoder(
            "popluation_centers_geocoder",
            geocoder_bounds,
            run_bounds=False,
            run_points=False,
            run_centres=True,
        )
    )
    tasks.append(UTMCoordinatesExtractor("utm_coordinate_extractor"))
    tasks.append(
        StatePlaneExtractor(
            "state_plain_coordinate_extractor",
            state_plane_lookup_filename,
            state_plane_zone_filename,
            state_code_filename,
        )
    )
    tasks.append(OutlierFilter("utm-outliers"))
    tasks.append(UTMStatePlaneFilter("utm-state-plane"))
    tasks.append(PointGeocoder("geocoded-georeferencing", ["point", "population"], 10))
    tasks.append(CornerPointExtractor("corner_point_extractor"))
    tasks.append(InferenceCoordinateExtractor("coordinate-inference"))
    tasks.append(ScaleExtractor("scale_extractor", ""))
    tasks.append(CreateGroundControlPoints("gcp_creator", create_random_pts=False))
    tasks.append(GeoReference("geo_referencer", 1))
    return Pipeline("wally-finder", "wally-finder", outputs, tasks)
