import logging
from typing import List, Optional, Tuple
from tasks.geo_referencing.georeference import QueryPoint

#
# Utility functions for loading / setting input parameters for the georeferencing pipeline
#

# --- lat/lon limits for USA (incl Alaska, Puerto Rico, etc.)
LON_LIMITS_US = (-180.0, -64.5)
LAT_LIMITS_US = (17.5, 71.6)
# --- lat/lon limits for the whole world
LON_LIMITS_WORLD = (-180.0, 180.0)
LAT_LIMITS_WORLD = (-90, 90.0)

logger = logging.getLogger(__name__)


def get_geofence_defaults(
    lon_limits: Tuple[float, float] = LON_LIMITS_US,
    lat_limits: Tuple[float, float] = LAT_LIMITS_US,
):
    """
    Get the default geo-fence ranges
    """
    lon_minmax = [min(lon_limits), max(lon_limits)]
    lat_minmax = [min(lat_limits), max(lat_limits)]
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
