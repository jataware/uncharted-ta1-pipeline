import os
import logging

from pathlib import Path

from pipelines.geo_referencing.output import (
    OutputCreator,
    GCPOutput,
    GeoReferencingOutput,
    IntegrationOutput,
    UserLeverOutput,
    SummaryOutput,
)
from tasks.common.pipeline import Pipeline
from tasks.common.task import TaskInput
from tasks.geo_referencing.coordinates_extractor import (
    GeoCoordinatesExtractor,
)
from tasks.geo_referencing.state_plane_extractor import StatePlaneExtractor
from tasks.geo_referencing.utm_extractor import UTMCoordinatesExtractor
from tasks.geo_referencing.filter import NaiveFilter, OutlierFilter, UTMStatePlaneFilter
from tasks.geo_referencing.geo_fencing import GeoFencer
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.geocode import Geocoder as rfGeocoder
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
from tasks.geo_referencing.inference import InferenceCoordinateExtractor
from tasks.geo_referencing.roi_extractor import (
    EntropyROIExtractor,
    ModelROIExtractor,
    buffer_fixed,
    buffer_image_ratio,
    buffer_roi_ratio,
)
from tasks.metadata_extraction.geocoder import Geocoder, NominatimGeocoder
from tasks.metadata_extraction.metadata_extraction import MetadataExtractor, LLM
from tasks.metadata_extraction.scale import ScaleExtractor
from tasks.metadata_extraction.text_filter import FilterMode, TextFilter
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor

from typing import List

logger = logging.getLogger("factory")


def run_step(input: TaskInput) -> bool:
    lats = input.get_data("lats", [])
    lons = input.get_data("lons", [])

    lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lats.items()))
    lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lons.items()))
    num_keypoints = min(len(lons_distinct), len(lats_distinct))
    logger.info(f"running step due to insufficient key points: {num_keypoints < 2}")
    return num_keypoints < 2


def create_geo_referencing_pipelines(
    extract_metadata: bool,
    output_dir: str,
    working_dir: str,
    segmentation_model_path: str,
    state_plane_lookup_filename: str,
    state_plane_zone_filename: str,
    state_code_filename: str,
    country_code_filename: str,
    ocr_gamma_correction: float,
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

    p = []

    tasks = []
    tasks.append(ResizeTextExtractor("first", Path(text_cache), False, True, 6000))
    tasks.append(EntropyROIExtractor("entropy roi"))
    if extract_metadata:
        tasks.append(MetadataExtractor("metadata_extractor", LLM.GPT_3_5_TURBO))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(UTMCoordinatesExtractor("fourth"))
    tasks.append(CreateGroundControlPoints("sixth"))
    tasks.append(GeoReference("seventh", 1))
    """p.append(
        Pipeline(
            "resize",
            "resize",
            [
                GeoReferencingOutput("geo"),
                SummaryOutput("summary"),
                UserLeverOutput("levers"),
                GCPOutput("gcps"),
                IntegrationOutput("schema"),
            ],
            tasks,
        )
    )"""

    tasks = []
    tasks.append(
        TileTextExtractor("first", Path(text_cache), 6000, gamma_correction=0.5)
    )
    tasks.append(EntropyROIExtractor("entropy roi"))
    if extract_metadata:
        tasks.append(
            MetadataExtractor(
                "metadata_extractor", LLM.GPT_3_5_TURBO, cache_dir=metadata_cache
            )
        )
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(UTMCoordinatesExtractor("fourth"))
    if extract_metadata:
        tasks.append(
            TextFilter(
                "map_area_filter",
                FilterMode.INCLUDE,
                "map_area_filter",
                ["map"],
                run_step,
            )
        )
        tasks.append(
            MetadataExtractor(
                "metadata_map_area_extractor",
                LLM.GPT_3_5_TURBO,
                "map_area_filter",
                run_step,
            )
        )
        tasks.append(
            Geocoder(
                "geo-places",
                geocoder_points,
                run_bounds=False,
                run_points=True,
            )
        )
        tasks.append(rfGeocoder("geocoded-georeferencing", ["point", "population"]))
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(CreateGroundControlPoints("sixth"))
    tasks.append(GeoReference("seventh", 1))
    """p.append(
        Pipeline(
            "tile",
            "tile",
            [
                GeoReferencingOutput("geo"),
                SummaryOutput("summary"),
                UserLeverOutput("levers"),
                GCPOutput("gcps"),
                IntegrationOutput("schema"),
            ],
            tasks,
        )
    )"""

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
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_fixed,
        )
    )
    if extract_metadata:
        tasks.append(
            TextFilter(
                "text_filter",
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
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    if extract_metadata:
        tasks.append(
            TextFilter(
                "map_area_filter",
                FilterMode.INCLUDE,
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
            )
        )
        tasks.append(
            Geocoder(
                "geo-centres",
                geocoder_bounds,
                run_bounds=False,
                run_points=False,
                run_centres=True,
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
        tasks.append(rfGeocoder("geocoded-georeferencing", ["point", "population"]))
    tasks.append(InferenceCoordinateExtractor("coordinate-inference"))
    tasks.append(ScaleExtractor("scaler", ""))
    tasks.append(CreateGroundControlPoints("seventh"))
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
                IntegrationOutput("schema"),
            ],
            tasks,
        )
    )

    tasks = []
    tasks.append(
        TileTextExtractor("first", Path(text_cache), 6000, gamma_correction=0.5)
    )
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            segmentation_cache,
            confidence_thres=0.25,
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_image_ratio,
        )
    )
    if extract_metadata:
        tasks.append(TextFilter("text_filter", output_key="filtered_ocr_text"))
        tasks.append(
            MetadataExtractor(
                "metadata_extractor",
                LLM.GPT_3_5_TURBO,
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
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    if extract_metadata:
        tasks.append(
            TextFilter(
                "map_area_filter",
                FilterMode.INCLUDE,
                "map_area_filter",
                ["map"],
                run_step,
            )
        )
        tasks.append(
            MetadataExtractor(
                "metadata_map_area_extractor",
                LLM.GPT_3_5_TURBO,
                "map_area_filter",
                run_step,
            )
        )
        tasks.append(
            Geocoder(
                "geo-places",
                geocoder_points,
                run_bounds=False,
                run_points=True,
                run_centres=False,
            )
        )
        tasks.append(
            Geocoder(
                "geo-centres",
                geocoder_bounds,
                run_bounds=False,
                run_points=False,
                run_centres=True,
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
        tasks.append(rfGeocoder("geocoded-georeferencing", ["point", "population"]))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    """p.append(
        Pipeline(
            "roi poly image",
            "roi poly",
            [
                GeoReferencingOutput("geo"),
                SummaryOutput("summary"),
                UserLeverOutput("levers"),
                GCPOutput("gcps"),
                IntegrationOutput("schema"),
            ],
            tasks,
        )
    )"""

    tasks = []
    tasks.append(TileTextExtractor("first", Path(text_cache), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            segmentation_cache,
            confidence_thres=0.25,
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_roi_ratio,
        )
    )
    if extract_metadata:
        tasks.append(TextFilter("text_filter", output_key="filtered_ocr_text"))
        tasks.append(
            MetadataExtractor(
                "metadata_extractor",
                LLM.GPT_3_5_TURBO,
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
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    if extract_metadata:
        tasks.append(
            TextFilter(
                "map_area_filter",
                FilterMode.INCLUDE,
                "map_area_filter",
                ["map"],
                run_step,
            )
        )
        tasks.append(
            MetadataExtractor(
                "metadata_map_area_extractor",
                LLM.GPT_3_5_TURBO,
                "map_area_filter",
                run_step,
            )
        )
        tasks.append(
            Geocoder(
                "geo-places",
                geocoder_points,
                run_bounds=False,
                run_points=True,
                run_centres=False,
            )
        )
        tasks.append(
            Geocoder(
                "geo-centres",
                geocoder_bounds,
                run_bounds=False,
                run_points=False,
                run_centres=True,
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
        tasks.append(rfGeocoder("geocoded-georeferencing", ["point", "population"]))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    """p.append(
        Pipeline(
            "roi poly roi",
            "roi poly",
            [
                GeoReferencingOutput("geo"),
                SummaryOutput("summary"),
                UserLeverOutput("levers"),
                GCPOutput("gcps"),
                IntegrationOutput("schema"),
            ],
            tasks,
        )
    )"""

    return p


def create_geo_referencing_pipeline(
    segmentation_model_path: str,
    outputs: List[OutputCreator],
    working_dir: str,
    country_code_filename: str,
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
        TileTextExtractor("first", Path(text_cache), 6000, gamma_correction=0.5)
    )
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            segmentation_cache,
            confidence_thres=0.25,
        )
    )
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_fixed,
        )
    )
    tasks.append(TextFilter("text_filter", output_key="filtered_ocr_text"))
    tasks.append(
        MetadataExtractor(
            "metadata_extractor",
            LLM.GPT_3_5_TURBO,
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
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    tasks.append(
        TextFilter(
            "map_area_filter",
            FilterMode.INCLUDE,
            "map_area_filter",
            ["map"],
            run_step,
        )
    )
    tasks.append(
        MetadataExtractor(
            "metadata_map_area_extractor",
            LLM.GPT_3_5_TURBO,
            "map_area_filter",
            run_step,
        )
    )
    tasks.append(
        Geocoder(
            "geo-places",
            geocoder_points,
            run_bounds=False,
            run_points=True,
            run_centres=False,
        )
    )
    tasks.append(
        Geocoder(
            "geo-centres",
            geocoder_bounds,
            run_bounds=False,
            run_points=False,
            run_centres=True,
        )
    )
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(OutlierFilter("utm-outliers"))
    tasks.append(rfGeocoder("geocoded-georeferencing", ["point", "population"]))
    tasks.append(ScaleExtractor("scaler", ""))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    return Pipeline("wally-finder", "wally-finder", outputs, tasks)
