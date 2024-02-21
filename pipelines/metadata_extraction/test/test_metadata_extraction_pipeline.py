from pathlib import Path

from pipelines.metadata_extraction.metadata_extraction_pipeline import (
    GeopackageIntegrationOutput,
)
from schema.ta1_schema import ProvenanceType
from tasks.common.pipeline import GeopackageOutput, PipelineResult
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)


class TestGeopackageIntegrationOutput:
    def test_create_output(self):
        # Create a mock PipelineResult
        metadata_extraction = MetadataExtraction(
            map_id="test_id",
            title="test_title",
            authors=["test_author_1", "test_author_2"],
            year="2021",
            scale="1:1000",
            quadrangles=["test_quadrangle_1", "test_quadrangle_2"],
            datum="test_datum",
            vertical_datum="test_vertical_datum",
            projection="test_projection",
            coordinate_systems=["test_coordinate_system_1", "test_coordinate_system_2"],
            base_map="test_base_map",
            counties=["test_county_1", "test_county_2"],
            states=["test_state_1", "test_state_2"],
            country="test_country",
            places=[],
        )
        pipeline_result = PipelineResult()
        pipeline_result.pipeline_id = "test_id"
        pipeline_result.pipeline_name = "test_name"
        pipeline_result.raster_id = "test_raster_id"
        pipeline_result.data[METADATA_EXTRACTION_OUTPUT_KEY] = metadata_extraction

        # Create an instance of GeopackageIntegrationOutput
        geopackage_path = Path("pipelines/metadata_extraction/test")
        geopackage_file = geopackage_path.joinpath(
            f"{pipeline_result.raster_id}_metadata_extraction.gpkg"
        )
        output_creator = GeopackageIntegrationOutput("output_id", str(geopackage_path))

        try:
            # Call the create_output method
            output = output_creator.create_output(pipeline_result)

            # Assert that the GeopackageOutput has the correct attributes
            assert output.pipeline_id == pipeline_result.pipeline_id
            assert output.pipeline_name == pipeline_result.pipeline_name

            # open the geopackage database
            assert isinstance(output, GeopackageOutput)

            gpk = output.data
            assert gpk.model is not None

            gpk_map = gpk.session.query(gpk.model.map).first()
            assert gpk_map.id == pipeline_result.raster_id  # type: ignore
            assert gpk_map.name == pipeline_result.raster_id  # type: ignore
            assert gpk_map.source_url == ""  # type: ignore
            assert gpk_map.image_url == ""  # type: ignore
            assert gpk_map.image_width == (  # type: ignore
                pipeline_result.image.width if pipeline_result.image else -1
            )
            assert gpk_map.image_height == (  # type: ignore
                pipeline_result.image.height if pipeline_result.image else -1
            )

            gpk_metadata = gpk.session.query(gpk.model.map_metadata).first()
            assert gpk_metadata.id == metadata_extraction.map_id  # type: ignore
            assert gpk_metadata.map_id == pipeline_result.raster_id  # type: ignore
            assert gpk_metadata.provenance == "modelled"  # type: ignore
            assert gpk_metadata.authors == ",".join(metadata_extraction.authors)  # type: ignore
            assert gpk_metadata.publisher == ""  # type: ignore
            assert gpk_metadata.confidence == 0.5  # type: ignore
            assert gpk_metadata.year == int(metadata_extraction.year)  # type: ignore
            assert gpk_metadata.scale == metadata_extraction.scale  # type: ignore
            assert gpk_metadata.title == metadata_extraction.title  # type: ignore

        finally:
            # delete the file
            geopackage_file.unlink()
