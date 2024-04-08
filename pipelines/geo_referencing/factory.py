import os

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
from tasks.geo_referencing.utm_extractor import UTMCoordinatesExtractor
from tasks.geo_referencing.filter import NaiveFilter, OutlierFilter
from tasks.geo_referencing.geo_fencing import GeoFencer
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.geocode import Geocoder as rfGeocoder
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
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


def run_step(input: TaskInput) -> bool:
    lats = input.get_data("lats", [])
    lons = input.get_data("lons", [])
    num_keypoints = min(len(lons), len(lats))
    print(f"running step due to insufficient key points: {num_keypoints < 2}")
    return num_keypoints < 2


def create_geo_referencing_pipelines(
    extract_metadata: bool, output_dir: str
) -> List[Pipeline]:
    p = []

    tasks = []
    tasks.append(
        ResizeTextExtractor("first", Path("temp/text/cache"), False, True, 6000)
    )
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
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(EntropyROIExtractor("entropy roi"))
    if extract_metadata:
        tasks.append(MetadataExtractor("metadata_extractor", LLM.GPT_3_5_TURBO))
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
                NominatimGeocoder(10, 5),
                run_bounds=False,
                run_points=True,
            )
        )
        tasks.append(rfGeocoder("geocoded-georeferencing"))
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
                IntegrationModelOutput("model"),
            ],
            tasks,
        )
    )"""

    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_xsection_20231201",
            "temp/segmentation/cache",
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
        tasks.append(TextFilter("text_filter", output_key="filtered_ocr_text"))
        tasks.append(
            MetadataExtractor(
                "metadata_extractor", LLM.GPT_3_5_TURBO, "filtered_ocr_text"
            )
        )
        tasks.append(
            Geocoder(
                "geo-bounds",
                NominatimGeocoder(10, 1),
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
                NominatimGeocoder(10, 5),
                run_bounds=False,
                run_points=True,
                run_centres=False,
            )
        )
        tasks.append(
            Geocoder(
                "geo-centres",
                NominatimGeocoder(10, 1),
                run_bounds=False,
                run_points=False,
                run_centres=True,
            )
        )
        tasks.append(UTMCoordinatesExtractor("fifth"))
        tasks.append(rfGeocoder("geocoded-georeferencing"))
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
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_xsection_20231201",
            "temp/segmentation/cache",
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
                "metadata_extractor", LLM.GPT_3_5_TURBO, "filtered_ocr_text"
            )
        )
        tasks.append(
            Geocoder(
                "geo-bounds",
                NominatimGeocoder(10, 1),
                run_bounds=True,
                run_points=False,
            )
        )
        tasks.append(GeoFencer("geofence"))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
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
                NominatimGeocoder(10, 5),
                run_bounds=False,
                run_points=True,
            )
        )
        tasks.append(rfGeocoder("geocoded-georeferencing"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
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
                IntegrationModelOutput("model"),
            ],
            tasks,
        )
    )"""

    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_xsection_20231201",
            "temp/segmentation/cache",
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
                "metadata_extractor", LLM.GPT_3_5_TURBO, "filtered_ocr_text"
            )
        )
        tasks.append(
            Geocoder(
                "geo-bounds",
                NominatimGeocoder(10, 1),
                run_bounds=True,
                run_points=False,
            )
        )
        tasks.append(GeoFencer("geofence"))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(NaiveFilter("fun"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
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
                NominatimGeocoder(10, 5),
                run_bounds=False,
                run_points=True,
            )
        )
        tasks.append(rfGeocoder("geocoded-georeferencing"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
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
                IntegrationModelOutput("model"),
            ],
            tasks,
        )
    )"""

    return p


def create_geo_referencing_pipeline(
    segmentation_model_path: str, outputs: List[OutputCreator]
) -> Pipeline:
    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            segmentation_model_path,
            "temp/segmentation/cache",
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
        MetadataExtractor("metadata_extractor", LLM.GPT_3_5_TURBO, "filtered_ocr_text")
    )
    tasks.append(
        Geocoder(
            "geo-bounds",
            NominatimGeocoder(10, 1),
            run_bounds=True,
            run_points=False,
        )
    )
    tasks.append(GeoFencer("geofence"))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
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
            NominatimGeocoder(10, 5),
            run_bounds=False,
            run_points=True,
        )
    )
    tasks.append(rfGeocoder("geocoded-georeferencing"))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    return Pipeline("wally-finder", "wally-finder", outputs, tasks)
