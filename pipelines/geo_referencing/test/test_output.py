import io
import pytest
from PIL import Image
from tasks.common.pipeline import BytesOutput, ImageOutput, PipelineResult
from tasks.geo_referencing.entities import (
    CORNER_POINTS_OUTPUT_KEY,
    CRS_OUTPUT_KEY,
    QUERY_POINTS_OUTPUT_KEY,
    GroundControlPoint as LARAGroundControlPoint,
)
from pipelines.geo_referencing.output import ProjectedMapOutput


@pytest.fixture
def pipeline_result():
    # Create a sample pipeline result with image, gcps, and crs
    image = Image.new("RGB", (100, 100))

    # write increasing pixel values to the image
    for i in range(100):
        for j in range(100):
            image.putpixel((i, j), (i, j, 0))

    gcps = [
        LARAGroundControlPoint(
            id="gcp1",
            pixel_x=10,
            pixel_y=10,
            latitude=40,
            longitude=-120,
            confidence=1.0,
        ),
        LARAGroundControlPoint(
            id="gcp2",
            pixel_x=90,
            pixel_y=10,
            latitude=40,
            longitude=-110,
            confidence=1.0,
        ),
        LARAGroundControlPoint(
            id="gcp3",
            pixel_x=90,
            pixel_y=50,
            latitude=30,
            longitude=-110,
            confidence=1.0,
        ),
        LARAGroundControlPoint(
            id="gcp4",
            pixel_x=10,
            pixel_y=50,
            latitude=30,
            longitude=-120,
            confidence=1.0,
        ),
    ]
    crs = "EPSG:4267"
    result = PipelineResult()
    result.pipeline_id = "pipeline_id"
    result.pipeline_name = "pipeline_name"
    result.image = image
    result.data = {
        QUERY_POINTS_OUTPUT_KEY: gcps,
        CRS_OUTPUT_KEY: crs,
        CORNER_POINTS_OUTPUT_KEY: gcps,
    }
    return result


def test_projected_map_output_creation(pipeline_result: PipelineResult):
    # Test if the ProjectedMapOutput is created successfully
    output = ProjectedMapOutput("output_id")
    result = output.create_output(pipeline_result)
    assert isinstance(result, BytesOutput)
    assert result.pipeline_id == "pipeline_id"
    assert result.pipeline_name == "pipeline_name"
    assert isinstance(result.data, io.BytesIO)
