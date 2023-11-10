
from pipelines.geo_referencing.output import (DetailedOutput, GCPOutput, GeoReferencingOutput, IntegrationOutput, UserLeverOutput, SummaryOutput)
from pipelines.geo_referencing.pipeline import Pipeline
from tasks.geo_referencing.coordinates_extractor import (GeocodeCoordinatesExtractor, GeoCoordinatesExtractor, UTMCoordinatesExtractor)
from tasks.geo_referencing.georeference import GeoReference
from tasks.geo_referencing.roi_extractor import (EntropyROIExtractor, ModelROIExtractor, buffer_fixed, buffer_image_ratio, buffer_roi_ratio)
from tasks.geo_referencing.task import Task
from tasks.geo_referencing.text_extraction import ResizeTextExtractor, TileTextExtractor

def create_geo_referencing_pipelines() -> list[Pipeline]:
    p = []

    tasks = []
    tasks.append(ResizeTextExtractor('first'))
    tasks.append(EntropyROIExtractor('entropy roi'))
    tasks.append(GeoCoordinatesExtractor('third'))
    tasks.append(UTMCoordinatesExtractor('fourth'))
    tasks.append(GeocodeCoordinatesExtractor('fifth'))
    tasks.append(GeoReference('sixth', 1))
    p.append(Pipeline('resize', 'resize', [GeoReferencingOutput('geo'), DetailedOutput('detailed'), SummaryOutput('summary'), UserLeverOutput('levers'), GCPOutput('gcps'), IntegrationOutput('schema')], tasks))

    tasks = []
    tasks.append(TileTextExtractor('first'))
    tasks.append(EntropyROIExtractor('entropy roi'))
    tasks.append(GeoCoordinatesExtractor('third'))
    tasks.append(UTMCoordinatesExtractor('fourth'))
    tasks.append(GeocodeCoordinatesExtractor('fifth'))
    tasks.append(GeoReference('sixth', 1))
    p.append(Pipeline('tile', 'tile', [GeoReferencingOutput('geo'), DetailedOutput('detailed'), SummaryOutput('summary'), UserLeverOutput('levers'), GCPOutput('gcps'), IntegrationOutput('schema')], tasks))

    tasks = []
    tasks.append(TileTextExtractor('first'))
    #tasks.append(ModelROIExtractor('model roi', buffer_fixed, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(ModelROIExtractor('model roi', buffer_fixed, '/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025'))
    tasks.append(GeoCoordinatesExtractor('third'))
    tasks.append(UTMCoordinatesExtractor('fourth'))
    tasks.append(GeocodeCoordinatesExtractor('fifth'))
    tasks.append(GeoReference('sixth', 1))
    p.append(Pipeline('roi poly fixed', 'roi poly', [GeoReferencingOutput('geo'), DetailedOutput('detailed'), SummaryOutput('summary'), UserLeverOutput('levers'), GCPOutput('gcps'), IntegrationOutput('schema')], tasks))

    tasks = []
    tasks.append(TileTextExtractor('first'))
    #tasks.append(ModelROIExtractor('model roi', buffer_image_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(ModelROIExtractor('model roi', buffer_image_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025'))
    tasks.append(GeoCoordinatesExtractor('third'))
    tasks.append(UTMCoordinatesExtractor('fourth'))
    tasks.append(GeocodeCoordinatesExtractor('fifth'))
    tasks.append(GeoReference('sixth', 1))
    p.append(Pipeline('roi poly image', 'roi poly', [GeoReferencingOutput('geo'), DetailedOutput('detailed'), SummaryOutput('summary'), UserLeverOutput('levers'), GCPOutput('gcps'), IntegrationOutput('schema')], tasks))

    tasks = []
    tasks.append(TileTextExtractor('first'))
    #tasks.append(ModelROIExtractor('model roi', buffer_roi_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/map_legend_segmentation_labels/ch1_validation_evaluation_labels_coco.json'))
    tasks.append(ModelROIExtractor('model roi', buffer_roi_ratio, '/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025'))
    tasks.append(GeoCoordinatesExtractor('third'))
    tasks.append(UTMCoordinatesExtractor('fourth'))
    tasks.append(GeocodeCoordinatesExtractor('fifth'))
    tasks.append(GeoReference('sixth', 1))
    p.append(Pipeline('roi poly roi', 'roi poly', [GeoReferencingOutput('geo'), DetailedOutput('detailed'), SummaryOutput('summary'), UserLeverOutput('levers'), GCPOutput('gcps'), IntegrationOutput('schema')], tasks))

    return p

def create_geo_referencing_pipeline() -> Pipeline:
    tasks = []
    tasks.append(TileTextExtractor('ocr'))
    tasks.append(ModelROIExtractor('model roi', buffer_fixed, '/Users/phorne/projects/criticalmaas/data/challenge_1/legend_and_map_segmentation_results_20231025'))
    tasks.append(GeoCoordinatesExtractor('geo'))
    tasks.append(UTMCoordinatesExtractor('utm'))
    tasks.append(GeocodeCoordinatesExtractor('geocode'))
    tasks.append(GeoReference('reference', 1))
    return Pipeline('wally-finder', 'wally-finder', [IntegrationOutput('schema')], tasks)
    