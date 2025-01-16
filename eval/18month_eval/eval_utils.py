import math
from geopy.distance import geodesic
from rasterio.transform import from_gcps, AffineTransformer
from typing import List
import re


def score_query_points(gcps) -> float:
    """
    score the query points
    - use the raster_io based transform to convert pixels to world coords
    - and then use a CRS_transform (if needed), to convert from source to target CRS systems

    Calc the RMSE geodesic error for all points
    """

    affine_transform = from_gcps(gcps)
    transformer = AffineTransformer(affine_transform)

    sum_sq_error = 0.0
    num_pts = 0
    for gcp in gcps:

        lonlat = transformer.xy(gcp.row, gcp.col)

        # lonlat = rio_transform.xy(gcp.row, gcp.col)
        #
        # if crs_transform:
        #     lonlat_new = crs_transform.transform(lonlat[0], lonlat[1])
        #     lonlat = lonlat_new
        latlon_gtruth = (gcp.y, gcp.x)
        err_dist = geodesic(latlon_gtruth, (lonlat[1], lonlat[0])).km

        sum_sq_error += err_dist * err_dist
        num_pts += 1

    if num_pts == 0:
        return -1

    rmse = math.sqrt(sum_sq_error / num_pts)
    return rmse


def point_property_for_grouping(props: List[str]) -> str:
    """
    choose an appropriate property for grouping point features from a shapefile
    (ie if a single shapefile contains different point symbol types)
    """
    # bad_list = ["dip", "strike", "angle", "id"]
    # ranked list of 'good' point type property labels
    good_list = [
        "type",
        "ftr_type",
        "pointtype",
        "pttype",
        "ptype",
        "stype",
        "feature_type",
        "descriptio",
        "description",
        "feature",
    ]

    if not props:
        print("No shapefile properties to group by")
        return ""

    print(props)
    for g in good_list:
        for p in props:
            p_norm = p.lower().strip()
            if p_norm == g:
                print(f"Grouping point features by property: {p}")
                return p

    return ""


# from https://stackoverflow.com/questions/1007481/how-to-replace-whitespaces-with-underscore
def urlify(s):

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", "-", s)

    return s
