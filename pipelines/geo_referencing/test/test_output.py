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
from tasks.geo_referencing.georeference import QueryPoint


@pytest.fixture
def pipeline_result():
    # Create a sample pipeline result with image, gcps, and crs
    image = Image.new("RGB", (100, 100))

    # write increasing pixel values to the image
    for i in range(100):
        for j in range(100):
            image.putpixel((i, j), (i, j, 0))

    gcps = [
        QueryPoint(
            id="gcp1",
            xy=(10, 10),
            lonlat_gtruth=None,
            confidence=1.0,
        ),
        QueryPoint(
            id="gcp2",
            xy=(90, 10),
            lonlat_gtruth=None,
            confidence=1.0,
        ),
        QueryPoint(
            id="gcp3",
            xy=(90, 50),
            lonlat_gtruth=None,
            confidence=1.0,
        ),
        QueryPoint(
            id="gcp4",
            xy=(10, 50),
            lonlat_gtruth=None,
            confidence=1.0,
        ),
    ]
    gcps[0].lonlat = (40, -120)
    gcps[1].lonlat = (40, -110)
    gcps[2].lonlat = (30, -110)
    gcps[3].lonlat = (30, -120)

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
