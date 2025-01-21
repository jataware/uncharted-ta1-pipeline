import json
import logging
import os
from io import BytesIO
from typing import List, Optional
import requests
import time

import httpx
import pika.spec as spec
from pydantic import BaseModel
from pika.adapters.blocking_connection import BlockingChannel as Channel

from schema.cdr_schemas.feature_results import FeatureResults
from schema.cdr_schemas.georeference import GeoreferenceResults
from schema.cdr_schemas.metadata import CogMetaData
from schema.mappers.cdr import GeoreferenceMapper, get_mapper
from tasks.common.image_cache import ImageCache
from tasks.common.io import BytesIOFileWriter, ImageFileReader, JSONFileWriter
from tasks.common.request_client import OutputType, RequestResult
from tasks.common.result_subscriber import LaraResultSubscriber
from tasks.geo_referencing.entities import (
    GeoreferenceResult as LARAGeoreferenceResult,
    GroundControlPoint as LARAGroundControlPoint,
)
from tasks.metadata_extraction.entities import MetadataExtraction as LARAMetadata
from tasks.point_extraction.entities import PointLabels as LARAPoints
from tasks.segmentation.entities import MapSegmentation as LARASegmentation
from tasks.geo_referencing.util import cps_to_transform, project_image

logger = logging.getLogger("write_result_subscriber")


class WriteResultSubscriber(LaraResultSubscriber):
    def __init__(
        self,
        result_queue,
        cdr_host,
        cdr_token,
        output,
        workdir,
        imagedir,
        host="localhost",
        port=5672,
        vhost="/",
        uid="",
        pwd="",
        metrics_url="",
    ):
        super().__init__(
            result_queue,
            cdr_host,
            cdr_token,
            host=host,
            port=port,
            vhost=vhost,
            uid=uid,
            pwd=pwd,
        )
        self._metrics_url = metrics_url
        self._output = output
        self._workdir = workdir
        self._image_cache = ImageCache(imagedir)
        self._image_cache._init_cache()

    def _process_lara_result(
        self,
        channel: Channel,
        method: spec.Basic.Deliver,
        properties: spec.BasicProperties,
        body: bytes,
    ):
        """
        Process the received LARA result by mapping it to the CDR schema and uploading it.

        Args:
            channel (Channel): The channel object.
            method (spec.Basic.Deliver): The method object.
            properties (spec.BasicProperties): The properties object.
            body (bytes): The body of the message.

        Returns:
            None

        Raises:
            Exception: If there is an error processing the result.
        """
        try:
            logger.info("received data from result channel")
            # parse the result
            body_decoded = json.loads(body.decode())
            result = RequestResult.model_validate(body_decoded)
            logger.info(
                f"processing result for request {result.id} for {result.image_id} of type {result.output_type}"
            )

            # add metric of job starting
            if self._metrics_url != "":
                requests.post(self._metrics_url + "/counter/writer_started?step=1")

            start_time = time.perf_counter()
            match result.output_type:
                case OutputType.SEGMENTATION:
                    logger.info("segmentation results received")
                    self._push_segmentation(result)
                case OutputType.METADATA:
                    logger.info("metadata results received")
                    self._push_metadata(result)
                case OutputType.POINTS:
                    logger.info("points results received")
                    self._push_points(result)
                case OutputType.GEOREFERENCING:
                    logger.info("georeferencing results received")
                    self._push_georeferencing(result)
                case _:
                    logger.info("unsupported output type received from queue")
            elasped_time = time.perf_counter() - start_time
            if self._metrics_url != "":
                requests.post(self._metrics_url + "/counter/writer_completed?step=1")
                requests.post(
                    self._metrics_url + "/histogram/writer?value=" + str(elasped_time)
                )

        except Exception as e:
            logger.exception(e)
            if self._metrics_url != "":
                requests.post(self._metrics_url + "/counter/writer_errored?step=1")

        logger.info("result processing finished")

    def _write_cdr_result(
        self,
        image_id: str,
        output_type: OutputType,
        result: BaseModel,
        image_bytes: Optional[BytesIO] = None,
    ):
        """
        Write the CDR result to a JSON file and optionally write image bytes to a file.

        Args:
            image_id (str): The ID of the image.
            output_type (OutputType): The type of output.
            result (BaseModel): The result to be written.
            image_bytes (Optional[BytesIO]): The image bytes to be written, if any.

        Returns:
            None
        """
        if self._output:
            writer = JSONFileWriter()
            output_file = os.path.join(
                self._output,
                f"{image_id}_{output_type.name.lower()}.json",
            )
            writer.process(output_file, result)

        if image_bytes:
            writer = BytesIOFileWriter()
            output_file = os.path.join(
                self._output,
                f"{image_id}_{output_type.name.lower()}.tif",
            )
            writer.process(output_file, image_bytes)

    def _push_georeferencing(self, result: RequestResult):
        # reproject image to file on disk for pushing to CDR
        georef_result_raw = json.loads(result.output)

        # don't write when the result is empty
        if len(georef_result_raw) == 0:
            logger.info(
                f"empty georef result received for {result.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[GeoreferenceResults] = None
        files_ = []
        image_bytes = None
        try:
            lara_result = LARAGeoreferenceResult.model_validate(georef_result_raw)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.GEOREFERENCE_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.GEOREFERENCE_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
            assert cdr_result is not None
            assert cdr_result.georeference_results is not None
            assert cdr_result.georeference_results[0] is not None
            assert cdr_result.georeference_results[0].projections is not None
            projection = cdr_result.georeference_results[0].projections[0]
            gcps = cdr_result.gcps
            output_file_name = projection.file_name
            output_file_name_full = os.path.join(
                self._workdir, "projected_image", output_file_name
            )

            assert gcps is not None
            lara_gcps = [
                LARAGroundControlPoint(
                    id=f"gcp.{i}",
                    pixel_x=gcp.px_geom.columns_from_left,
                    pixel_y=gcp.px_geom.rows_from_top,
                    latitude=gcp.map_geom.latitude if gcp.map_geom.latitude else 0,
                    longitude=gcp.map_geom.longitude if gcp.map_geom.longitude else 0,
                    confidence=gcp.confidence if gcp.confidence else 0,
                )
                for i, gcp in enumerate(gcps)
            ]

            logger.info(
                f"projecting image {result.image_path} to {output_file_name_full}. CRS: {lara_result.crs} -> {GeoreferenceMapper.DEFAULT_OUTPUT_CRS}"
            )
            image_bytes = self._project_georeference(
                result.image_id,
                result.image_path,
                lara_result.crs,
                GeoreferenceMapper.DEFAULT_OUTPUT_CRS,
                lara_gcps,
            )
            # pass the image bytes to the CDR
            files_.append(("files", (output_file_name, image_bytes)))
        except Exception as e:
            logger.exception(
                "formatting for CDR schema failed for {result.request.image_id}: {e}",
            )

            # create an empty result to send to cdr
            cdr_result = GeoreferenceResults(
                cog_id=result.image_id,
                georeference_results=[],
                gcps=[],
                system=self.PIPELINE_SYSTEM_NAMES[self.GEOREFERENCE_PIPELINE],
                system_version=self.PIPELINE_SYSTEM_VERSIONS[
                    self.GEOREFERENCE_PIPELINE
                ],
            )

        assert cdr_result is not None
        try:
            # write the result to disk if output is set
            if self._output:
                self._write_cdr_result(
                    result.image_id, result.output_type, cdr_result, image_bytes
                )
                return

            # push the result to CDR
            logger.info(f"pushing result for request {result.id} to CDR")
            headers = {"Authorization": f"Bearer {self._cdr_token}"}
            client = httpx.Client(follow_redirects=True)
            resp = client.post(
                f"{self._cdr_host}/v1/maps/publish/georef",
                data={"georef_result": json.dumps(cdr_result.model_dump())},
                files=files_,
                headers=headers,
                timeout=None,
            )
            logger.info(
                f"result for request {result.id} sent to CDR with response {resp.status_code}: {resp.content}"
            )
        except Exception as e:
            logger.exception(
                f"error when attempting to submit georeferencing results: {e}"
            )
            raise e

    def _project_georeference(
        self,
        source_image_id: str,
        source_image_url: str,
        source_crs: str,
        target_crs: str,
        gcps: List[LARAGroundControlPoint],
    ) -> Optional[BytesIO]:
        """
        Projects an image to a new coordinate reference system using ground control points.

        Args:
            source_image_path (str): The path to the source image.
            target_image_path (str): The path to the target image.
            target_crs (str): The target coordinate reference system.
            gcps (List[GroundControlPoint]): The ground control points.
        """
        # open the image
        image = self._image_cache.fetch_cached_result(f"{source_image_id}.tif")
        if not image:
            # not cached - download from s3 and cache - we assume no credentials are needed
            # on the download url
            logger.info(f"cache miss - downloading image from {source_image_url}")
            image_file_reader = ImageFileReader()
            image = image_file_reader.process(source_image_url, anonymous=True)
            if image is None:
                logger.error(f"failed to download image from {source_image_url}")
                return
            self._image_cache.write_result_to_cache(image, f"{source_image_id}.tif")

        # create the transform and use it to project the image
        geo_transform = cps_to_transform(gcps, source_crs, target_crs)
        image_bytes = project_image(image, geo_transform, target_crs)

        # write the projected image out
        return image_bytes

    def _push_features(self, result: RequestResult, model: FeatureResults):
        """
        Pushes the features result to the CDR
        """
        try:
            if self._output:
                self._write_cdr_result(result.image_id, result.output_type, model)
                return

            logger.info(f"pushing features result for request {result.id} to CDR")
            headers = {
                "Authorization": f"Bearer {self._cdr_token}",
                "Content-Type": "application/json",
            }
            client = httpx.Client(follow_redirects=True)
            resp = client.post(
                f"{self._cdr_host}/v1/maps/publish/features",
                data=model.model_dump_json(),  #   type: ignore
                headers=headers,
                timeout=None,
            )
            logger.info(
                f"result for request {result.id} sent to CDR with response {resp.status_code}: {resp.content}"
            )
        except Exception as e:
            logger.exception(f"error when attempting to submit feature results: {e}")
            raise e

    def _push_segmentation(self, result: RequestResult):
        """
        Pushes the segmentation result to the CDR
        """
        segmentation_raw_result = json.loads(result.output)

        # don't write when the result is empty
        if len(segmentation_raw_result) == 0:
            logger.info(
                f"empty segmentation result received for {result.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[FeatureResults] = None
        try:
            lara_result = LARASegmentation.model_validate(segmentation_raw_result)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.SEGMENTATION_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.SEGMENTATION_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except Exception as e:
            logger.exception(
                f"mapping segmentation to CDR schema failed for {result.image_id}: {e}",
            )
            return

        assert cdr_result is not None
        self._push_features(result, cdr_result)

    def _push_points(self, result: RequestResult):
        points_raw_result = json.loads(result.output)

        # don't write when the result is empty
        if len(points_raw_result) == 0:
            logger.info(
                f"empty point result received for {result.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[FeatureResults] = None
        try:
            lara_result = LARAPoints.model_validate(points_raw_result)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.POINTS_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.POINTS_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except Exception as e:
            logger.exception(
                f"mapping points to CDR schema failed for {result.image_id}: {e}",
            )
            return

        assert cdr_result is not None
        self._push_features(result, cdr_result)

    def _push_metadata(self, result: RequestResult):
        """
        Pushes the metadata result to the CDR
        """
        metadata_result_raw = json.loads(result.output)

        # don't write when the result is empty
        if len(metadata_result_raw) == 0:
            logger.info(
                f"empty metadata result received for {result.image_id} - skipping send"
            )
            return

        # validate the result by building the model classes
        cdr_result: Optional[CogMetaData] = None
        try:
            lara_result = LARAMetadata.model_validate(metadata_result_raw)
            mapper = get_mapper(
                lara_result,
                self.PIPELINE_SYSTEM_NAMES[self.METADATA_PIPELINE],
                self.PIPELINE_SYSTEM_VERSIONS[self.METADATA_PIPELINE],
            )
            cdr_result = mapper.map_to_cdr(lara_result)  #   type: ignore
        except Exception as e:
            logger.exception(
                f"mapping metadata to CDR schema failed for {result.image_id}: {e}",
            )
            return

        assert cdr_result is not None

        # wrap metadata into feature result
        final_result = FeatureResults(
            cog_id=result.image_id,
            cog_metadata_extractions=[cdr_result],
            system=cdr_result.system,
            system_version=cdr_result.system_version,
        )

        self._push_features(result, final_result)
