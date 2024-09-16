import os
import logging

from pathlib import Path

from pipelines.geo_referencing.output import (
    GeoreferencingOutput,
    GCPOutput,
    ScoringOutput,
    ProjectedMapOutput,
    UserLeverOutput,
    SummaryOutput,
)
from tasks.common.pipeline import Pipeline
from tasks.common.task import Task, TaskInput
from tasks.geo_referencing.coordinates_extractor import (
    GeoCoordinatesExtractor,
)
from tasks.geo_referencing.corner_point_extractor import CornerPointExtractor
from tasks.geo_referencing.entities import (
    GEOREFERENCING_OUTPUT_KEY,
    LEVERS_OUTPUT_KEY,
    PROJECTED_MAP_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    SCORING_OUTPUT_KEY,
    SUMMARY_OUTPUT_KEY,
)
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


class GeoreferencingPipeline(Pipeline):
    def __init__(
        self,
        working_dir: str,
        segmentation_model_path: str,
        state_plane_lookup_filename: str,
        state_plane_zone_filename: str,
        state_code_filename: str,
        country_code_filename: str,
        ocr_gamma_correction: float,
        gpu_enabled: bool,
    ):
        geocoding_cache_bounds = os.path.join(
            working_dir, "geocoding_cache_bounds.json"
        )

        geocoding_cache_points = os.path.join(
            working_dir, "geocoding_cache_points.json"
        )
        geocoder_bounds = NominatimGeocoder(
            10, geocoding_cache_bounds, 1, country_code_filename=country_code_filename
        )
        geocoder_points = NominatimGeocoder(
            10, geocoding_cache_points, 5, country_code_filename=country_code_filename
        )

        segmentation_cache = os.path.join(working_dir, "segmentation")
        text_cache = os.path.join(working_dir, "text")
        metadata_cache = os.path.join(
            working_dir, f"metadata-gamma-{ocr_gamma_correction}"
        )
        geocoder_thresh = 10

        tasks: List[Task] = [
            TileTextExtractor(
                "first", Path(text_cache), 6000, gamma_correction=ocr_gamma_correction
            ),
            DetectronSegmenter(
                "segmenter",
                segmentation_model_path,
                segmentation_cache,
                confidence_thres=0.25,
                gpu=gpu_enabled,
            ),
            ModelROIExtractor(
                "model roi",
                buffer_fixed,
            ),
            TextFilter(
                "text_filter",
                # input_key="metadata_ocr",
                output_key="filtered_ocr_text",
                classes=[
                    "cross_section",
                    "legend_points_lines",
                    "legend_polygons",
                ],
            ),
            MetadataExtractor(
                "metadata_extractor",
                LLM.GPT_4_O,
                "filtered_ocr_text",
                cache_dir=metadata_cache,
            ),
            Geocoder(
                "geocoded_geobounds",
                geocoder_bounds,
                run_bounds=True,
                run_points=False,
                run_centres=False,
            ),
            GeoFencer("geofence"),
            GeoCoordinatesExtractor("geo_coordinates_extractor"),
            ROIFilter("roi_filter"),
            DistinctDegreeOutlierFilter("uniqueness_filter"),
            HighQualityCoordinateFilter("quality_filter"),
            OutlierFilter("outlier filter"),
            NaiveFilter("naive_filter"),
            TextFilter(
                "map_area_filter",
                FilterMode.INCLUDE,
                TEXT_EXTRACTION_OUTPUT_KEY,
                "map_area_filter",
                ["map"],
                self._run_step,
            ),
            MetadataExtractor(
                "metadata_map_area_extractor",
                LLM.GPT_4_O,
                "map_area_filter",
                self._run_step,
                include_place_bounds=True,
            ),
            Geocoder(
                "geo-places",
                geocoder_points,
                run_bounds=False,
                run_points=True,
                run_centres=False,
                should_run=self._run_step,
            ),
            Geocoder(
                "geo-centres",
                geocoder_bounds,
                run_bounds=False,
                run_points=False,
                run_centres=True,
                should_run=self._run_step,
            ),
            UTMCoordinatesExtractor("utm-_coordinates"),
            StatePlaneExtractor(
                "state_plane_coordinates",
                state_plane_lookup_filename,
                state_plane_zone_filename,
                state_code_filename,
            ),
            OutlierFilter("utm-outliers"),
            UTMStatePlaneFilter("utm-state-plane"),
            PointGeocoder(
                "geocoded-georeferencing", ["point", "population"], geocoder_thresh
            ),
            BoxGeocoder("geocoded-box", ["point", "population"], geocoder_thresh),
            CornerPointExtractor("corner_point_extractor"),
            InferenceCoordinateExtractor("coordinate-inference"),
            ScaleExtractor("scaler", ""),
            CreateGroundControlPoints("gcp_creation", create_random_pts=False),
            GeoReference("georeference", 1),
        ]

        outputs = [
            ScoringOutput(SCORING_OUTPUT_KEY),
            SummaryOutput(SUMMARY_OUTPUT_KEY),
            UserLeverOutput(LEVERS_OUTPUT_KEY),
            GCPOutput(QUERY_POINTS_OUTPUT_KEY),
            ProjectedMapOutput(PROJECTED_MAP_OUTPUT_KEY),
            GeoreferencingOutput(GEOREFERENCING_OUTPUT_KEY),
        ]

        super().__init__(
            "georeferencing_roi_poly_fixed", "Georeferencing", outputs, tasks
        )

    def _run_step(self, input: TaskInput) -> bool:
        """
        Determines whether or not a step should run based on the identified key points

        Args:
            input (TaskInput): The input to the task

        Returns:
            bool: True if the step should be run
        """
        lats = input.get_data("lats", {})
        lons = input.get_data("lons", {})

        lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lats.items()))
        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lons.items()))
        num_keypoints = min(len(lons_distinct), len(lats_distinct))
        logger.info(f"running step due to insufficient key points: {num_keypoints < 2}")
        return num_keypoints < 2
