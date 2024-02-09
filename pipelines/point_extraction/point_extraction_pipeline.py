import logging
from pathlib import Path
from typing import List
from tasks.point_extraction.point_extractor import YOLOPointDetector
from tasks.point_extraction.point_orientation_extractor import PointOrientationExtractor
from tasks.point_extraction.tiling import Tiler, Untiler
from tasks.point_extraction.entities import MapImage
from tasks.common.pipeline import (
    BaseModelOutput,
    BaseModelListOutput,
    Pipeline,
    PipelineResult,
    OutputCreator,
    Output,
    GeopackageOutput,
)
from criticalmaas.ta1_geopackage import GeopackageDatabase
from tasks.text_extraction.text_extractor import ResizeTextExtractor
from tasks.segmentation.detectron_segmenter import DetectronSegmenter
from schema.ta1_schema import (
    PointFeature,
    PointType,
    ProvenanceType,
)


logger = logging.getLogger(__name__)


class PointExtractionPipeline(Pipeline):
    """
    Pipeline for extracting map point symbols, orientation, and their associated orientation and magnitude values, if present

    Args:
        model_path: path to point symbol extraction model weights
        model_path_segmenter: path to segmenter model weights
        work_dir: cache directory
    """

    def __init__(
        self,
        model_path: str,
        model_path_segmenter: str,
        work_dir: str,
        verbose=False,
    ):
        # extract text from image, segmentation to only keep the map area,
        # tile, extract points, untile, predict direction
        logger.info("Initializing Point Extraction Pipeline")
        tasks = []
        tasks.append(
            # TODO: could use Tiled Text extractor here? ... better recall for dip magnitude extraction?
            ResizeTextExtractor(
                "resize_text",
                Path(work_dir).joinpath("text"),
                to_blocks=True,
                document_ocr=False,
                pixel_lim=6000,
            )
        )
        if model_path_segmenter:
            tasks.append(
                DetectronSegmenter(
                    "detectron_segmenter",
                    model_path_segmenter,
                    str(Path(work_dir).joinpath("segmentation")),
                )
            )
        else:
            logger.warning(
                "Not using image segmentation. 'model_path_segmenter' param not given"
            )
        tasks.extend(
            [
                Tiler("tiling"),
                YOLOPointDetector(
                    "point_detection", model_path, work_dir, batch_size=20
                ),
                Untiler("untiling"),
                PointOrientationExtractor("point_orientation_extraction"),
            ]
        )

        outputs = [
            MapPointLabelOutput("map_point_label_output"),
            IntegrationOutput("integration_output"),
        ]

        super().__init__("point_extraction", "Point Extraction", outputs, tasks)
        self._verbose = verbose


class MapPointLabelOutput(OutputCreator):
    def __init__(self, id: str):
        super().__init__(id)

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a MapPointLabel object from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            MapPointLabel: The map point label extraction object.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])
        return BaseModelOutput(
            pipeline_result.pipeline_id,
            pipeline_result.pipeline_name,
            map_image,
        )


class IntegrationOutput(OutputCreator):
    """
    OutputCreator for point extraction pipeline.
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
        Validates the point extraction pipeline result and converts into the TA1 schema representation

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            Output: The output of the pipeline.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])

        point_features: List[PointFeature] = []
        if map_image.labels:
            for i, map_pt_label in enumerate(map_image.labels):

                # TODO - add in optional point class description text? (eg extracted from legend?)
                pt_type = PointType(
                    id=str(map_pt_label.class_id),
                    name=map_pt_label.class_name,
                    description=None,
                )
                # centre pixel of point bounding box
                xy_centre = (
                    int((map_pt_label.x2 + map_pt_label.x1) / 2),
                    int((map_pt_label.y2 + map_pt_label.y1) / 2),
                )

                pt_feature = PointFeature(
                    id=str(i),  # TODO - use a better unique ID here?
                    type=pt_type,
                    map_geom=None,  # TODO - add in real-world coords if/when geo-ref transform available
                    px_geom=xy_centre,
                    dip_direction=map_pt_label.direction,
                    dip=map_pt_label.dip,
                    confidence=map_pt_label.score,
                    provenance=ProvenanceType.modelled,
                )
                point_features.append(pt_feature)

        result = BaseModelListOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, point_features
        )
        return result


class GeopackageIntegrationOutput(OutputCreator):
    def __init__(self, id: str, file_path: Path):
        super().__init__(id)
        self._file_path = file_path

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        """
        Creates a geopackage output from the pipeline result.

        Args:
            pipeline_result (PipelineResult): The pipeline result.

        Returns:
            GeopackageOutput: A geopackage containing the points extraction results.
        """
        map_image = MapImage.model_validate(pipeline_result.data["map_image"])

        # for pixel co-ords results -- use as default for now
        db = GeopackageDatabase(str(self._file_path), crs="CRITICALMAAS:0")
        # for world co-ords results
        # db = GeopackageDatabase(str(self._file_path), crs="EPSG:4326")

        if db.model is None:
            raise ValueError("db.model is None")

        if not map_image.labels:
            # no point extraction results for this image
            return GeopackageOutput(
                pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
            )

        # TODO: these db write calls will throw an exception if records already exist. Ideally should check if records exist or overwrite?

        db.write_models(
            [
                # map
                db.model.map(
                    id=pipeline_result.raster_id,
                    name=pipeline_result.raster_id,
                    source_url="",
                    image_url="",
                    image_width=(
                        pipeline_result.image.width if pipeline_result.image else -1
                    ),
                    image_height=(
                        pipeline_result.image.height if pipeline_result.image else -1
                    ),
                ),
            ]
        )

        db.write_models(
            [
                # point types
                db.model.point_type(
                    id="strike_and_dip",
                    name="bedding",
                    description=" strike/dip or inclined bedding",
                ),
                db.model.point_type(
                    id="horizontal_bedding",
                    name="bedding",
                    description="horizontal bedding",
                ),
                db.model.point_type(
                    id="overturned_bedding",
                    name="bedding",
                    description="overturned bedding",
                ),
                db.model.point_type(
                    id="vertical_bedding",
                    name="bedding",
                    description="vertical bedding",
                ),
                db.model.point_type(
                    id="inclined_foliation",
                    name="foliation",
                    description="inclined foliation",
                ),
                db.model.point_type(
                    id="vertical_foliation",
                    name="foliation",
                    description="vertical foliation",
                ),
                db.model.point_type(
                    id="vertical_joint", name="joint", description="vertical joint"
                ),
                db.model.point_type(
                    id="sink_hole", name="other", description="sink hole"
                ),
                db.model.point_type(
                    id="gravel_borrow_pit",
                    name="other",
                    description="gravel borrow bit",
                ),
                db.model.point_type(
                    id="mine_shaft", name="other", description="mine shaft"
                ),
                db.model.point_type(
                    id="prospect", name="other", description="prospect"
                ),
                db.model.point_type(
                    id="mine_tunnel", name="other", description="mine tunnel or adit"
                ),
                db.model.point_type(
                    id="mine_quarry", name="other", description="mine quarry"
                ),
            ]
        )

        point_features = []
        for i, map_pt_label in enumerate(map_image.labels):

            # (NOTE: using extraction ID = <raster_id>_points_<i>
            extr_id = f"{pipeline_result.raster_id}_points_{i}"

            # centre pixel of point bounding box
            xy_centre = (
                int((map_pt_label.x2 + map_pt_label.x1) / 2),
                int((map_pt_label.y2 + map_pt_label.y1) / 2),
            )
            # TODO -- we are writing point locations in pixel coords
            # geopackage currently doesn't support writing both pixel and real-world coords

            pt_feat = {
                "properties": {
                    "id": extr_id,
                    "map_id": pipeline_result.raster_id,
                    "type": map_pt_label.class_name,
                    "confidence": map_pt_label.score,
                    "provenance": "modelled",
                    "dip_direction": map_pt_label.direction,
                    "dip": map_pt_label.dip,
                },
                "geometry": {"type": "Point", "coordinates": xy_centre},
            }
            point_features.append(pt_feat)

        if point_features:
            db.write_features("point_feature", point_features)

        return GeopackageOutput(
            pipeline_result.pipeline_id, pipeline_result.pipeline_name, db
        )
