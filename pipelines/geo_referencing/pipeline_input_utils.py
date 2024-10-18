import logging
from typing import List, Optional, Tuple
from tasks.geo_referencing.georeference import QueryPoint

#
# Utility functions for loading / setting input parameters for the georeferencing pipeline
#

GEOFENCE_DEFAULTS = {
    # --- lat/lon limits for the whole world
    "world": {"lon": (-180.0, 180.0), "lat": (-90.0, 90.0)},
    # --- lat/lon limits for USA (incl Alaska, Puerto Rico, etc.)
    "us": {"lon": (-180.0, -64.5), "lat": (17.5, 71.6)},
}

logger = logging.getLogger(__name__)


def get_geofence_defaults(geofence_region: str = "world") -> Tuple:
    """
    Get the default geo-fence ranges

    geofence_region: current excepted values are "world" or "us"
    """
    if geofence_region not in GEOFENCE_DEFAULTS:
        logger.warning(
            f"geofence_region {geofence_region} not found; using whole world as default geofence"
        )
        geofence_region = "world"

    lon_minmax = GEOFENCE_DEFAULTS[geofence_region]["lon"]
    lat_minmax = GEOFENCE_DEFAULTS[geofence_region]["lat"]
    lon_sign_factor = 1.0
    return (lon_minmax, lat_minmax, lon_sign_factor)


def parse_query_file(
    csv_query_file: str, image_size: Optional[Tuple[float, float]] = None
) -> List[QueryPoint]:
    """
    Expected schema is of the form:
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,8250,12796,-105.72065081057087,43.40255034572461
    ...
    Note: NAD83* columns may not be present
    row (y) and col (x) = pixel coordinates to query
    NAD83* = (if present) are ground truth answers (lon and lat) for the query x,y pt
    """

    first_line = True
    x_idx = 2
    y_idx = 1
    lon_idx = 3
    lat_idx = 4
    query_pts = []
    try:
        with open(csv_query_file) as f_in:
            for line in f_in:
                if line.startswith("raster_") or first_line:
                    first_line = False
                    continue  # header line, skip

                rec = line.split(",")
                if len(rec) < 3:
                    continue
                raster_id = rec[0]
                x = int(rec[x_idx])
                y = int(rec[y_idx])
                if image_size is not None:
                    # sanity check that query points are not > image dimensions!
                    if x > image_size[0] or y > image_size[1]:
                        err_msg = (
                            "Query point {}, {} is outside image dimensions".format(
                                x, y
                            )
                        )
                        raise IOError(err_msg)
                lonlat_gt = None
                if len(rec) >= 5:
                    lon = float(rec[lon_idx])
                    lat = float(rec[lat_idx])
                    if lon != 0 and lat != 0:
                        lonlat_gt = (lon, lat)
                query_pts.append(QueryPoint(raster_id, (x, y), lonlat_gt))

    except Exception as e:
        logger.exception(f"EXCEPTION parsing query file: {str(e)}", exc_info=True)

    return query_pts
