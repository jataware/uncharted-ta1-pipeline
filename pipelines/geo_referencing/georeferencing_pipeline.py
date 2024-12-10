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
from tasks.common.task import EvaluateHalt, Task
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
    CoordSource,
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
from tasks.geo_referencing.point_geocoder import PointGeocoder
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
from tasks.geo_referencing.inference import InferenceCoordinateExtractor
from tasks.geo_referencing.roi_extractor import ROIExtractor
from tasks.geo_referencing.finalize_coordinates import FinalizeCoordinates
from tasks.metadata_extraction.entities import GeoPlaceType
from tasks.metadata_extraction.geocoder import Geocoder
from tasks.metadata_extraction.geocoding_service import NominatimGeocoder
from tasks.metadata_extraction.metadata_extraction import (
    LLM_PROVIDER,
    MetadataExtractor,
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
        model: str,
        api_version: str,
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
            4,
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
                metrics_url=metrics_url,
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
                api_version,
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
            ScaleAnalyzer("scale analyzer", dpi_default=300),
            # Extracts all the possible geo coordinates from the UNFILTERED text
            GeoCoordinatesExtractor("geo coordinates extractor"),
            # Filters out any coordinates that are not in the buffered region of interest (ie. around the outside of the map poly)
            ROIFilter("region of interest coord filter"),
            # Test for outliers in each of the X and Y directions independently using linear regression
            OutlierFilter(
                "lat/lon outlier filter", coord_source_check=[CoordSource.LAT_LON]
            ),
            # Infer addtional points in a given direction if there are insufficient points in that direction
            InferenceCoordinateExtractor("lat/lon inference"),
            # Extract corner points from the map area
            CornerPointExtractor("corner point extractor"),
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
            OutlierFilter(
                "UTM outlier filter",
                coord_source_check=[CoordSource.UTM, CoordSource.STATE_PLANE],
            ),
            # Filter out UTM or state plane coords if sufficient lat/lon coords are present
            UTMStatePlaneFilter("UTM / state plane coordinate filter"),
            # Infer addtional points in a given direction if there are insufficient points in that direction
            InferenceCoordinateExtractor("UTM inference"),
            # Geocode the places extracted from the map area
            Geocoder(
                "places / population centers geocoder",
                points_geocoder,
                run_bounds=False,
                run_points=True,
                run_centres=True,
            ),
            # Generate georeferencing keypoints from the place and population centre geo coordinates
            PointGeocoder(
                "point-based geocoder", [GeoPlaceType.POINT, GeoPlaceType.POPULATION]
            ),
            OutlierFilter(
                "geocoder outlier filter", coord_source_check=[CoordSource.GEOCODER]
            ),
            # Finalize coordinate extractions (ie checks for co-linear or ill-conditioned coord spacing)
            FinalizeCoordinates("finalize coordinates"),
            # Create ground control points for use in downstream tasks
            CreateGroundControlPoints("gcp creation", create_random_pts=False),
            # Run the final georeferencing step using either the regression-based inference method
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

        super().__init__("georeferencing", "Georeferencing", outputs, tasks)
