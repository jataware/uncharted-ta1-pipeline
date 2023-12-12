from pathlib import Path

from pipelines.geo_referencing.output import (
    GCPOutput,
    GeoReferencingOutput,
    IntegrationOutput,
    UserLeverOutput,
    SummaryOutput,
)
from tasks.common.pipeline import Pipeline
from tasks.geo_referencing.coordinates_extractor import (
    GeocodeCoordinatesExtractor,
    GeoCoordinatesExtractor,
    UTMCoordinatesExtractor,
)
from tasks.geo_referencing.filter import OutlierFilter
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.ground_control import CreateGroundControlPoints
from tasks.geo_referencing.roi_extractor import (
    EntropyROIExtractor,
    ModelROIExtractor,
    buffer_fixed,
    buffer_image_ratio,
    buffer_roi_ratio,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from tasks.text_extraction.text_extractor import ResizeTextExtractor, TileTextExtractor


def create_geo_referencing_pipelines() -> list[Pipeline]:
    p = []

    tasks = []
    tasks.append(
        ResizeTextExtractor("first", Path("temp/text/cache"), False, True, 6000)
    )
    tasks.append(EntropyROIExtractor("entropy roi"))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(UTMCoordinatesExtractor("fourth"))
    tasks.append(GeocodeCoordinatesExtractor("fifth"))
    tasks.append(CreateGroundControlPoints("sixth"))
    tasks.append(GeoReference("seventh", 1))
    p.append(
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
    )

    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(EntropyROIExtractor("entropy roi"))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(UTMCoordinatesExtractor("fourth"))
    tasks.append(GeocodeCoordinatesExtractor("fifth"))
    tasks.append(CreateGroundControlPoints("sixth"))
    tasks.append(GeoReference("seventh", 1))
    p.append(
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
    )

    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_20230",
            "temp/segmentation/cache",
            confidence_thres=0.25,
        )
    )
    # tasks.append(ModelROIExtractor('model roi', buffer_fixed, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_fixed,
            "/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025",
        )
    )
    # tasks.append(ModelROIExtractor('model roi', buffer_fixed, '/Users/phorne/projects/criticalmaas/data/challenge_1/quick-seg'))
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(GeocodeCoordinatesExtractor("sixth"))
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
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_20230",
            "temp/segmentation/cache",
            confidence_thres=0.25,
        )
    )
    # tasks.append(ModelROIExtractor('model roi', buffer_image_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_image_ratio,
            "/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025",
        )
    )
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(GeocodeCoordinatesExtractor("sixth"))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    p.append(
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
    )

    tasks = []
    tasks.append(TileTextExtractor("first", Path("temp/text/cache"), 6000))
    tasks.append(
        DetectronSegmenter(
            "segmenter",
            "https://s3.t1.uncharted.software/lara/models/segmentation/layoutlmv3_20230",
            "temp/segmentation/cache",
            confidence_thres=0.25,
        )
    )
    # tasks.append(ModelROIExtractor('model roi', buffer_roi_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_roi_ratio,
            "/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025",
        )
    )
    tasks.append(GeoCoordinatesExtractor("third"))
    tasks.append(OutlierFilter("fourth"))
    tasks.append(UTMCoordinatesExtractor("fifth"))
    tasks.append(GeocodeCoordinatesExtractor("sixth"))
    tasks.append(CreateGroundControlPoints("seventh"))
    tasks.append(GeoReference("eighth", 1))
    p.append(
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
    )

    return p


def create_geo_referencing_pipeline() -> Pipeline:
    tasks = []
    tasks.append(TileTextExtractor("ocr", Path("temp/text/cache"), 6000))
    tasks.append(
        ModelROIExtractor(
            "model roi",
            buffer_fixed,
            "/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025",
        )
    )
    tasks.append(GeoCoordinatesExtractor("geo"))
    tasks.append(UTMCoordinatesExtractor("utm"))
    tasks.append(GeocodeCoordinatesExtractor("geocode"))
    tasks.append(GeoReference("reference", 1))
    return Pipeline(
        "wally-finder", "wally-finder", [IntegrationOutput("schema")], tasks
    )
