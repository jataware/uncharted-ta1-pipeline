import os
import glob

from geopy.distance import distance as geo_distance
from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipelines
from pipelines.geo_referencing.output import CSVWriter, JSONWriter
from tasks.common.pipeline import PipelineInput
from tasks.geo_referencing.georeference import QueryPoint
from util.coordinates import absolute_minmax
from util.json import read_json_file

from typing import Union

FOV_RANGE_KM = (
    700  # [km] max range of a image's field-of-view (around the clue coord pt)
)
LON_MINMAX = [-66.0, -180.0]  # fallback geo-fence (ALL of USA + Alaska)
LAT_MINMAX = [24.0, 73.0]

CLUE_PATH_IN = ""
QUERY_PATH_IN = ""
POINTS_PATH_IN = ""
IMG_FILE_EXT = "tif"
CLUE_FILEN_SUFFIX = "_clue"

Image.MAX_IMAGE_PIXELS = 400000000
IMG_CACHE = "temp/images/"
GEOCODE_CACHE = "temp/geocode/"
os.makedirs(IMG_CACHE, exist_ok=True)
os.makedirs("temp/text/cache", exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


def create_input(
    raster_id: str, image_path: str, points_path: str, query_path: str, clue_path: str
):
    input = PipelineInput()
    input.image = Image.open(image_path)
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_params(clue_path, use_abs=False)
    input.params["lon_minmax"] = lon_minmax
    input.params["lat_minmax"] = lat_minmax
    input.params["lon_sign_factor"] = lon_sign_factor

    query_pts = query_points_from_points(raster_id, points_path)
    if not query_pts:
        query_pts = parse_query_file(query_path, input.image.size)
    input.params["query_pts"] = query_pts

    return input


def process_folder(image_folder: str):
    # get the pipelines
    pipelines = create_geo_referencing_pipelines()

    img_path_tmp = os.path.join(image_folder, "*." + IMG_FILE_EXT)
    image_filenames = glob.glob(img_path_tmp)
    image_filenames.sort()
    print(f"found {len(image_filenames)} images to process")
    if len(image_filenames) == 0:
        # check nested folders for images
        print("no images found, checking nested folders")
        img_path_tmp = os.path.join(image_folder, "*", "*." + IMG_FILE_EXT)
        image_filenames = glob.glob(img_path_tmp)
        image_filenames.sort()
        print(f"found {len(image_filenames)} nested images to process")

    results = {}
    results_summary = {}
    results_levers = {}
    results_gcps = {}
    results_integration = {}
    writer_csv = CSVWriter()
    writer_json = JSONWriter()
    for p in pipelines:
        results[p.id] = []
        results_summary[p.id] = []
        results_levers[p.id] = []
        results_gcps[p.id] = []
        results_integration[p.id] = []

    for image_path in image_filenames:
        print(f"processing {image_path}")
        image_base_filename = os.path.basename(image_path)
        image_base_filename = os.path.splitext(image_base_filename)[0]
        raster_id = os.path.splitext(image_path[len(image_folder) :])[0].replace(
            "/", "-"
        )

        clue_path = os.path.join(
            CLUE_PATH_IN, image_base_filename + CLUE_FILEN_SUFFIX + ".csv"
        )
        query_path = os.path.join(QUERY_PATH_IN, image_base_filename + ".csv")
        points_path = os.path.join(POINTS_PATH_IN, f"pipeline_output_{raster_id}.json")

        input = create_input(raster_id, image_path, points_path, query_path, clue_path)

        for pipeline in pipelines:
            print(f"running pipeline {pipeline.id}")
            output = pipeline.run(input)
            results[pipeline.id].append(output["geo"])
            results_summary[pipeline.id].append(output["summary"])
            results_levers[pipeline.id].append(output["levers"])
            results_gcps[pipeline.id].append(output["gcps"])
            results_integration[pipeline.id].append(output["schema"])
            print(f"done pipeline {pipeline.id}\n\n")

        for p in pipelines:
            writer_csv.output(results[p.id], {"path": f"output/test-{p.id}.csv"})
            writer_csv.output(
                results_summary[p.id], {"path": f"output/test_summary-{p.id}.csv"}
            )
            writer_json.output(
                results_levers[p.id], {"path": f"output/test_levers-{p.id}.json"}
            )
            writer_json.output(
                results_gcps[p.id], {"path": f"output/test_gcps-{p.id}.json"}
            )
            writer_json.output(
                results_integration[p.id], {"path": f"output/test_schema-{p.id}.json"}
            )

    for p in pipelines:
        writer_csv.output(results[p.id], {"path": f"output/test-{p.id}.csv"})
        writer_csv.output(
            results_summary[p.id], {"path": f"output/test_summary-{p.id}.csv"}
        )
        writer_json.output(
            results_levers[p.id], {"path": f"output/test_levers-{p.id}.json"}
        )
        writer_json.output(
            results_gcps[p.id], {"path": f"output/test_gcps-{p.id}.json"}
        )
        writer_json.output(
            results_integration[p.id], {"path": f"output/test_schema-{p.id}.json"}
        )


def get_geofence(
    csv_clue_file,
    fov_range_km,
    lon_limits=(-66.0, -180.0),
    lat_limits=(24.0, 73.0),
    use_abs=True,
):
    # parse clue CSV file
    (clue_lon, clue_lat, clue_ok) = parse_clue_file(csv_clue_file)
    if clue_ok:
        # if False:
        print("Using lon/lat clue: {}, {}".format(clue_lon, clue_lat))
        dist_km = (
            fov_range_km / 2.0
        )  # distance from clue pt in all directions (N,E,S,W)
        fov_pt_north = geo_distance(kilometers=dist_km).destination(
            (clue_lat, clue_lon), bearing=0
        )
        fov_pt_east = geo_distance(kilometers=dist_km).destination(
            (clue_lat, clue_lon), bearing=90
        )
        fov_degrange_lon = abs(fov_pt_east[1] - clue_lon)
        fov_degrange_lat = abs(fov_pt_north[0] - clue_lat)
        lon_minmax = [clue_lon - fov_degrange_lon, clue_lon + fov_degrange_lon]
        lat_minmax = [clue_lat - fov_degrange_lat, clue_lat + fov_degrange_lat]

    else:
        # if no lat/lon clue, fall-back to full geo-fence of USA + Alaska
        print("WARNING! No lon/lat clue found. Using full geo-fence for USA + Alaska")
        lon_minmax = lon_limits
        lat_minmax = lat_limits

    lon_sign_factor = 1.0

    return (absolute_minmax(lon_minmax), absolute_minmax(lat_minmax), lon_sign_factor)


def parse_query_file(csv_query_file, image_size=None):
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
        print("EXCEPTION parsing query file: {}".format(csv_query_file))
        print(e)

    # print('Num query points parsed: {}'.format(len(query_pts)))

    return query_pts


def query_points_from_points(
    raster_id: str, points_file: str
) -> Union[None, list[QueryPoint]]:
    return None
    if not os.path.isfile(points_file):
        return None

    query_points = []
    points_raw = read_json_file(points_file)
    for pt in points_raw["labels"]:
        x = int((pt["x1"] + pt["x2"]) / 2)
        y = int((pt["y1"] + pt["y2"]) / 2)
        query_points.append(
            QueryPoint(raster_id, (x, y), None, properties={"label": pt["class_name"]})
        )

    return query_points


def get_params(clue_path: str, use_abs: bool = True):
    return get_geofence(
        clue_path,
        fov_range_km=FOV_RANGE_KM,
        lon_limits=LON_MINMAX,
        lat_limits=LAT_MINMAX,
        use_abs=use_abs,
    )


def parse_clue_file(csv_clue_file):
    """
    Expected schema is of the form:
    raster_ID,NAD83_x,NAD83_y
    GEO_0004,-105.72065081057087,43.40255034572461

    Or possibly
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,,,-105.72065081057087,43.40255034572461
    """

    first_line = True
    got_clue = False
    lon = 0.0
    lat = 0.0
    try:
        with open(csv_clue_file) as f_in:
            for line in f_in:
                if line.startswith("raster_") or first_line:
                    first_line = False
                    continue  # header line, skip

                if got_clue:
                    break

                rec = line.split(",")
                if len(rec) < 3:
                    continue
                if len(rec) < 5:
                    lon = float(rec[1])
                    lat = float(rec[2])
                else:
                    lon = round(float(rec[3]), 1)  # round to 1 decimal place
                    lat = round(float(rec[4]), 1)
                got_clue = True
    except Exception as e:
        print("EXCEPTION parsing clue file!")
        print(e)

    return (lon, lat, got_clue)
