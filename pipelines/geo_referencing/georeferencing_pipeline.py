import logging

from pipelines.geo_referencing.output import (
    GeoreferencingOutput,
    GCPOutput,
    ScoringOutput,
    ProjectedMapOutput,
    UserLeverOutput,
    SummaryOutput,
)
from tasks.common.io import append_to_cache_location
from tasks.common.pipeline import OutputCreator, Pipeline
from tasks.common.task import EvaluateHalt, Task, TaskInput
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
from tasks.geo_referencing.outlier_filter import OutlierFilter
from tasks.geo_referencing.filter import (
    ROIFilter,
    UTMStatePlaneFilter,
)
from tasks.geo_referencing.geo_fencing import GeoFencer
from tasks.geo_referencing.scale_analyzer import ScaleAnalyzer
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.geocode import PointGeocoder, BoxGeocoder
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
from tasks.geo_referencing.inference import InferenceCoordinateExtractor
from tasks.geo_referencing.roi_extractor import ROIExtractor
from tasks.metadata_extraction.entities import GeoPlaceType
from tasks.metadata_extraction.geocoder import Geocoder
from tasks.metadata_extraction.geocoding_service import NominatimGeocoder
from tasks.metadata_extraction.metadata_extraction import (
    LLM_PROVIDER,
    MetadataExtractor,
    LLM,
)
from tasks.metadata_extraction.text_filter import (
    FilterMode,
    TextFilter,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.segmentation.segmenter_utils import map_missing
from tasks.text_extraction.text_extractor import TileTextExtractor

from typing import List

logger = logging.getLogger(__name__)


class GeoreferencingPipeline(Pipeline):
    def __init__(
        self,
        working_dir: str,
        segmentation_model_path: str,
        state_plane_lookup_filename: str,
        state_plane_zone_filename: str,
        state_code_filename: str,
        country_code_filename: str,
        geocoded_places_filename: str,
        ocr_gamma_correction: float,
        model: LLM,
        provider: LLM_PROVIDER,
        projected: bool,
        diagnostics: bool,
        gpu_enabled: bool,
        metrics_url: str = "",
    ):
        geocoding_cache_bounds = append_to_cache_location(
            working_dir, "geocoding_cache_bounds.json"
        )

        geocoding_cache_points = append_to_cache_location(
            working_dir, "geocoding_cache_points.json"
        )

        # Nominatim geocoder service for bounds
        bounds_geocoder = NominatimGeocoder(
            10,
            geocoding_cache_bounds,
            1,
            country_code_filename=country_code_filename,
            state_code_filename=state_code_filename,
            geocoded_places_filename=geocoded_places_filename,
        )

        # Nominatim geocoder service for points
        points_geocoder = NominatimGeocoder(
            10,
            geocoding_cache_points,
            5,
            country_code_filename=country_code_filename,
            state_code_filename=state_code_filename,
            geocoded_places_filename=geocoded_places_filename,
        )

        segmentation_cache = append_to_cache_location(working_dir, "segmentation")
        text_cache = append_to_cache_location(working_dir, "text")
        metadata_cache = append_to_cache_location(working_dir, "metadata")
        geocoder_thresh = 10

        tasks: List[Task] = [
            # Segments the image into map,legend and cross section
            DetectronSegmenter(
                "segmenter",
                segmentation_model_path,
                segmentation_cache,
                confidence_thres=0.25,
                gpu=gpu_enabled,
            ),
            # terminate the pipeline if the map region is not found - this will immediately return empty outputs
            EvaluateHalt(
                "map_presence_check",
                map_missing,
            ),
            # Extracts text from the tiled image
            TileTextExtractor(
                "tiled text extractor",
                text_cache,
                6000,
                gamma_correction=ocr_gamma_correction,
            ),
            # Defines an allowed region for cooredinates to occupy by buffering
            # the extracted map area polyline by a fixed amount
            ROIExtractor("region of interest extractor"),
            # Filters out any text that is in the cross section or the legend areas of the
            # map
            TextFilter(
                "metadata text filter",
                FilterMode.EXCLUDE,
                output_key="filtered_ocr_text",
                classes=[
                    "cross_section",
                    "legend_points_lines",
                    "legend_polygons",
                ],
                class_threshold=100,
            ),
            # Runs metadata extraction on the text remaining after the text filter above
            # is applied
            MetadataExtractor(
                "metadata_extractor",
                model,
                provider,
                "filtered_ocr_text",
                cache_location=metadata_cache,
                metrics_url=metrics_url,
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
            # Extract and analyze map scale info
            ScaleAnalyzer("scale analyzer", dpi=300),
            # Extracts all the possible geo coordinates from the UNFILTERED text
            GeoCoordinatesExtractor("geo coordinates extractor"),
            # Filters out any coordinates that are not in the buffered region of interest (ie. around the outside of the map poly)
            ROIFilter("region of interest coord filter"),
            # Test for outliers in each of the X and Y directions independently using linear regression
            OutlierFilter("lat/lon outlier filter"),
            # Geocode the places extracted from the map area
            Geocoder(
                "places / population centers geocoder",
                points_geocoder,
                run_bounds=False,
                run_points=True,
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
                [GeoPlaceType.POINT, GeoPlaceType.POPULATION],
                geocoder_thresh,
            ),
            # Generate georeferencing points from the extent of the place and population centre geo coordinates
            BoxGeocoder(
                "geocoded box transformation",
                [GeoPlaceType.POINT, GeoPlaceType.POPULATION],
                geocoder_thresh,
            ),
            # Extract corner points from the map area
            CornerPointExtractor("corner point extractor"),
            # Infer addtional points in a given direction if there are insufficient points in that direction
            InferenceCoordinateExtractor("coordinate inference"),
            # Create ground control points for use in downstream tasks
            CreateGroundControlPoints("gcp  creation", create_random_pts=False),
            # Run the final georeferencing step using either the regression-based inference method, or the corner point
            # methood if there are sufficient corner points
            GeoReference("georeference", 1),
        ]

        # assign the baseline georeferencing output
        outputs: List[OutputCreator] = [GeoreferencingOutput(GEOREFERENCING_OUTPUT_KEY)]

        # assign additional diagnostics outputs if requested
        if diagnostics:
            outputs.extend(
                [
                    ScoringOutput(SCORING_OUTPUT_KEY),
                    SummaryOutput(SUMMARY_OUTPUT_KEY),
                    UserLeverOutput(LEVERS_OUTPUT_KEY),
                    GCPOutput(QUERY_POINTS_OUTPUT_KEY),
                ]
            )
        # create the projected map output if requested
        if projected:
            outputs.append(ProjectedMapOutput(PROJECTED_MAP_OUTPUT_KEY))

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

        # TODO -- could filter on Coords with status OK?
        # lats = list(filter(lambda c: c._status == CoordStatus.OK, lats.values()))
        # lons = list(filter(lambda c: c._status == CoordStatus.OK, lons.values()))

        lats_distinct = set(map(lambda x: x[1].get_parsed_degree(), lats.items()))
        lons_distinct = set(map(lambda x: x[1].get_parsed_degree(), lons.items()))
        num_keypoints = min(len(lons_distinct), len(lats_distinct))
        logger.info(f"running step due to insufficient key points: {num_keypoints < 2}")
        return num_keypoints < 2
