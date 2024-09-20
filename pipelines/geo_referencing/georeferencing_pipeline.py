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
from tasks.common.io import append_to_cache_location
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

        # Nominatim geocoder service for bounds
        bounds_geocoder = NominatimGeocoder(
            10, geocoding_cache_bounds, 1, country_code_filename=country_code_filename
        )

        # Nominatim geocoder service for points
        points_geocoder = NominatimGeocoder(
            10, geocoding_cache_points, 5, country_code_filename=country_code_filename
        )

        segmentation_cache = append_to_cache_location(working_dir, "segmentation")
        text_cache = append_to_cache_location(working_dir, "text")
        metadata_cache = append_to_cache_location(working_dir, "metadata")
        geocoder_thresh = 10

        tasks: List[Task] = [
            # Extracts text from the tiled image
            TileTextExtractor(
                "tiled text extractor",
                text_cache,
                6000,
                gamma_correction=ocr_gamma_correction,
            ),
            # Segments the image into map,legend and cross section
            DetectronSegmenter(
                "segmenter",
                segmentation_model_path,
                segmentation_cache,
                confidence_thres=0.25,
                gpu=gpu_enabled,
            ),
            # Defines an allowed region for cooredinates to occupy by buffering
            # the extracted map area polyline by a fixed amount
            ModelROIExtractor(
                "fixed region of interest extractor",
                buffer_fixed,
            ),
            # Filters out any text that is in the cross section or the legend areas of the
            # map
            TextFilter(
                "metadata text filter",
                output_key="filtered_ocr_text",
                classes=[
                    "cross_section",
                    "legend_points_lines",
                    "legend_polygons",
                ],
            ),
            # Runs metadata extraction on the text remaining after the text filter above
            # is applied
            MetadataExtractor(
                "metadata_extractor",
                LLM.GPT_4_O,
                "filtered_ocr_text",
                cache_location=metadata_cache,
            ),
            # Creates geo locations for country and state of the map area based on the metadata
            Geocoder(
                "country / state geocoder",
                bounds_geocoder,
                run_bounds=True,
                run_points=False,
                run_centres=False,
            ),
            # Creates a geofence based on the country and state geocoded locations
            GeoFencer("country / state geofence"),
            # Extracts all the possible geo coordinates from the UNFILTERED text
            GeoCoordinatesExtractor("geo coordinates extractor"),
            # Filters out any coordinates that are not in the buffered region of interest (ie. around the outside of the map poly)
            ROIFilter("region of interest coord filter"),
            # Remove coordinates that are duplicates but don't appear to be part of the main map area (ie. from an inset map)
            DistinctDegreeOutlierFilter("uniqueness coord filter"),
            # Remove low confidence coordinates if there are high confidence coordinates nearby (in pixel space)
            HighQualityCoordinateFilter("quality coord filter"),
            # Test for outliers in each of the X and Y directions independently using linear regression
            OutlierFilter("regression outlier coord filter"),
            # Cluster based on geo locations and remove those that are outliers
            NaiveFilter("naive geo-space coord filter"),
            # Filter out any of the original text that is not inside the map area
            TextFilter(
                "map area text retainer",
                FilterMode.INCLUDE,
                TEXT_EXTRACTION_OUTPUT_KEY,
                "map_area_filter",
                ["map"],
                self._run_step,
            ),
            # Run metadata extraction on the map area text only
            MetadataExtractor(
                "metadata map area extractor",
                LLM.GPT_4_O,
                "map_area_filter",
                self._run_step,
                include_place_bounds=True,
            ),
            # Geocode the places extracted from the map area
            Geocoder(
                "place geocoder",
                points_geocoder,
                run_bounds=False,
                run_points=True,
                run_centres=False,
                should_run=self._run_step,
            ),
            # Geo code the population centres extracted from the map area
            Geocoder(
                "population center geocoder",
                bounds_geocoder,
                run_bounds=False,
                run_points=False,
                run_centres=True,
                should_run=self._run_step,
            ),
            # Exract any UTM coordinates that are present - UTM zone will be inferred
            UTMCoordinatesExtractor("utm coordinates extractor"),
            # Extract any state plane coordinates that are present
            StatePlaneExtractor(
                "state plane coordinate extractor",
                state_plane_lookup_filename,
                state_plane_zone_filename,
                state_code_filename,
            ),
            # Test UTM coords for outliers in each of the X and Y directions independently using linear regression
            OutlierFilter("UTM outlier filter"),
            # Filter out UTM or state plane coords if sufficient lat/lon coords are present
            UTMStatePlaneFilter("UTM / state plane coordinate filter"),
            # Generate georeferencing points from the full set of place and population centre geo coordinates
            PointGeocoder(
                "geocoded point transformation",
                ["point", "population"],
                geocoder_thresh,
            ),
            # Generate georeferencing points from the extent of the place and population centre geo coordinates
            BoxGeocoder(
                "geocoded box transformation", ["point", "population"], geocoder_thresh
            ),
            # Extract corner points from the map area
            CornerPointExtractor("corner point extractor"),
            # Infer addtional points in a given direction if there are insufficient points in that direction
            InferenceCoordinateExtractor("coordinate inference"),
            # This step doesn't seem to be doing anything given a lack of scale file being supplied
            # ScaleExtractor("scaler", ""),
            # Create ground control points for use in downstream tasks
            CreateGroundControlPoints("gcp  creation", create_random_pts=False),
            # Run the final georeferencing step using either the regression-based inference method, or the corner point
            # methood if there are sufficient corner points
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
