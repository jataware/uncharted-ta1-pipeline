from typing import List
from urllib.parse import urlparse
from pathlib import Path
import os

from schema.ta1_schema import PageExtraction, ExtractionIdentifier
from pipelines.segmentation.s3_data_cache import S3DataCache
from segmentation.entities import MapSegmentation
from tasks.common.pipeline import (
    Pipeline,
    PipelineResult,
    Output,
    OutputCreator,
    BaseModelOutput,
    BaseModelListOutput,
)
from tasks.segmentation.detectron_segmenter import DetectronSegmenter


class ModelPaths:
    def __init__(self, model_weights_path: Path, model_config_path: Path):
        self.model_weights_path = model_weights_path
        self.model_config_path = model_config_path


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

        model_paths = self._prep_config_data(model_data_path, model_data_cache_path)

        tasks = [
            DetectronSegmenter(
                "segmenter",
                str(model_paths.model_config_path),
                str(model_paths.model_weights_path),
                confidence_thres=confidence_thres,
            )
        ]

        outputs = [
            MapSegmentationOutput("map_segmentation_output"),
            IntegrationOutput("integration_output"),
        ]
        super().__init__("map-segmentation", "Map Segmentation", outputs, tasks)

    def _prep_config_data(
        self, model_data_path: str, data_cache_path: str
    ) -> ModelPaths:
        """
        prepare local data cache and download model weights, if needed

        Args:
            model_data_path (str): The path to the folder containing the model weights and config files
            data_cache_path (str): The path to the local data cache.

        Returns:
            ModelPaths: The paths to the model weights and config files.
        """

        local_model_data_path = None
        local_lm_config_path = None
        local_det_config_path = None

        # check if path is a URL

        if model_data_path.startswith("s3://") or model_data_path.startswith("http"):
            if data_cache_path == "":
                raise ValueError(
                    "'data_cache_path' must be specified when fetching model data from S3"
                )

            s3_host = ""
            s3_path = ""
            s3_bucket = ""

            res = urlparse(model_data_path)
            s3_host = res.scheme + "://" + res.netloc
            s3_path = res.path.lstrip("/")
            s3_bucket = s3_path.split("/")[0]
            s3_path = s3_path.lstrip(s3_bucket)
            s3_path = s3_path.lstrip("/")

            # create local data cache, if doesn't exist, and connect to S3
            s3_data_cache = S3DataCache(
                data_cache_path,
                s3_host,
                s3_bucket,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "<UNSET>"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "<UNSET>"),
            )

            # check for model weights and config files in the folder
            s3_subfolder = s3_path[: s3_path.rfind("/")]
            for s3_key in s3_data_cache.list_bucket_contents(s3_subfolder):
                if s3_key.endswith(".pth"):
                    local_model_data_path = Path(
                        s3_data_cache.fetch_file_from_s3(s3_key, overwrite=False)
                    )
                elif s3_key.endswith(".yaml"):
                    local_lm_config_path = Path(
                        s3_data_cache.fetch_file_from_s3(s3_key, overwrite=False)
                    )
                elif s3_key.endswith(".json"):
                    local_det_config_path = Path(
                        s3_data_cache.fetch_file_from_s3(s3_key, overwrite=False)
                    )
        else:
            # check for model weights and config files in the folder
            # iterate over files in folder
            for f in Path(model_data_path).iterdir():
                if f.is_file():
                    if f.suffix == ".pth":
                        local_model_data_path = f
                    elif f.suffix == ".yaml":
                        local_lm_config_path = f
                    elif f.suffix == ".json":
                        local_det_config_path = f

        # check that we have all the files we need
        if not local_model_data_path or not local_model_data_path.is_file():
            raise ValueError(f"Model weights file not found at {model_data_path}")

        if not local_det_config_path or not local_det_config_path.is_file():
            raise ValueError(f"Detectron config file not found at {model_data_path}")

        if not local_lm_config_path or not local_lm_config_path.is_file():
            raise ValueError(f"LayoutLM config file not found at {model_data_path}")

        return ModelPaths(local_model_data_path, local_lm_config_path)


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
