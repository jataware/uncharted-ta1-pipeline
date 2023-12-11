from typing import List

from schema.ta1_schema import PageExtraction, ExtractionIdentifier
from tasks.segmentation.entities import MapSegmentation
from tasks.common.pipeline import (
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
    BaseModelListOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter


class SegmentationPipeline(Pipeline):
    """
    Pipeline for segmenting maps into different components (map area, legend, etc.).
    """

    def __init__(
        self,
        model_data_path: str,
        model_data_cache_path: str = "",
        confidence_thres=0.25,
    ):
        """
        Initializes the pipeline.

        Args:
            config_path (str): The path to the Detectron2 config file.
            model_weights_path (str): The path to the Detectron2 model weights file.
            confidence_thres (float): The confidence threshold for the segmentation.
        """

        tasks = [
            DetectronSegmenter(
                "segmenter",
                model_data_path,
                model_data_cache_path,
                confidence_thres=confidence_thres,
            )
        ]

        outputs = [
            MapSegmentationOutput("map_segmentation_output"),
            IntegrationOutput("integration_output"),
        ]
        super().__init__("map-segmentation", "Map Segmentation", outputs, tasks)


class MapSegmentationOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult):
        """
        Creates a MapSegmentation object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MapSegmentation: The map segmentation extraction object.
        """
        map_segmentation = MapSegmentation.model_validate(pipeline_result.data)
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_segmentation,
        )


class IntegrationOutput(OutputCreator):
    """
    OutputCreator for text extraction pipeline.
    """

    def __init__(self, id):
        """
        Initializes the output creator.

        Args:
            id (str): The ID of the output creator.
        """
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Validates the pipeline result and converts into the TA1 schema representation

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_segmentation = MapSegmentation.model_validate(pipeline_result.data)

        page_extractions: List[PageExtraction] = []
        for segment in map_segmentation.segments:
            page_extraction = PageExtraction(
                name="segmentation",
                model=ExtractionIdentifier(
                    id=1, model=segment.id_model, field=segment.class_label
                ),
                ocr_text="",
                bounds=segment.poly_bounds,
                # confidence = segment.confidence, // TODO: add to schema
                color_estimation=None,
            )
            page_extractions.append(page_extraction)

        result = BaseModelListOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, page_extractions
        )
        return result
