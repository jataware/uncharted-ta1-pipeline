from typing import List
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
import rasterio as rio
import rasterio.transform as riot
from schema.cdr_schemas.georeference import GroundControlPoint
import json
import argparse
from tasks.geo_referencing.entities import GroundControlPoint as LARAGroundControlPoint


def project_georeference_output(
    source_image_path: str,
    target_image_path: str,
    target_crs: str,
    gcps: List[GroundControlPoint],
):
    # open the image
    img = Image.open(source_image_path)
    _, height = img.size

    # create the transform
    geo_transform = _cps_to_transform(gcps, height=height, to_crs=target_crs)

    # use the transform to project the image
    _project_image(source_image_path, target_image_path, geo_transform, target_crs)


def project_georeference(
    source_image_path: str,
    target_image_path: str,
    target_crs: str,
    gcps: List[GroundControlPoint],
):
    # open the image
    img = Image.open(source_image_path)
    _, height = img.size

    # create the transform
    geo_transform = _cps_to_transform(gcps, height=height, to_crs=target_crs)

    # use the transform to project the image
    _project_image(source_image_path, target_image_path, geo_transform, target_crs)


def _project_image(
    source_image_path: str,
    target_image_path: str,
    geo_transform: Affine,
    crs: str,
):
    with rio.open(source_image_path) as raw:
        bounds = riot.array_bounds(raw.height, raw.width, geo_transform)
        pro_transform, pro_width, pro_height = calculate_default_transform(
            crs,
            crs,
            raw.width,
            raw.height,
            *tuple(bounds),
            dst_width=raw.width,
            dst_height=raw.height
        )
        pro_kwargs = raw.profile.copy()
        pro_kwargs.update(
            {
                "driver": "COG",
                "crs": {"init": crs},
                "transform": pro_transform,
                "width": pro_width,
                "height": pro_height,
            }
        )
        _raw_data = raw.read()
        with rio.open(target_image_path, "w", **pro_kwargs) as pro:
            for i in range(raw.count):
                _ = reproject(
                    source=_raw_data[i],
                    destination=rio.band(pro, i + 1),
                    src_transform=geo_transform,
                    src_crs=crs,
                    dst_transform=pro_transform,
                    dst_crs=crs,
                    resampling=Resampling.bilinear,
                    num_threads=8,
                    warp_mem_limit=256,
                )


def _cps_to_transform(
    gcps: List[LARAGroundControlPoint], height: int, to_crs: str
) -> Affine:
    cps = [
        {
            "row": float(gcp.px_geom.rows_from_top),
            "col": float(gcp.px_geom.columns_from_left),
            "x": float(gcp.map_geom.longitude),  #   type: ignore
            "y": float(gcp.map_geom.latitude),  #   type: ignore
            "crs": gcp.crs,
        }
        for gcp in gcps
    ]
    cps_p = []
    for cp in cps:
        proj = Transformer.from_crs(cp["crs"], to_crs, always_xy=True)
        x_p, y_p = proj.transform(xx=cp["x"], yy=cp["y"])
        cps_p.append(
            riot.GroundControlPoint(row=cp["row"], col=cp["col"], x=x_p, y=y_p)
        )

    return riot.from_gcps(cps_p)


def _load_gcps(gcps_path: str) -> List[LARAGroundControlPoint]:
    with open(gcps_path, "r") as f:
        gcps = json.load(f)
        gcps_list = gcps["gcps"]
    return [GroundControlPoint(**gcp) for gcp in gcps_list]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to the source image")
    parser.add_argument("--output", type=str, help="Path to the target image")
    parser.add_argument("--gcps", type=str, help="Path to the GCPS JSON file")
    args = parser.parse_args()

    gcps = _load_gcps(args.gcps)
    project_georeference(args.input, args.output, "EPSG:4326", gcps)
