
import os
import glob

from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipelines
from pipelines.geo_referencing.output import (CSVWriter, JSONWriter)
from pipelines.geo_referencing.pipeline import PipelineInput
from tasks.geo_referencing.src.georeference import QueryPoint

FOV_RANGE_KM = 700              # [km] max range of a image's field-of-view (around the clue coord pt)
LON_MINMAX = [-66.0, -180.0]     # fallback geo-fence (ALL of USA + Alaska)
LAT_MINMAX = [24.0, 73.0]

QUERY_PATH_IN = '/Users/phorne/projects/criticalmaas/data/challenge_1/AI4CMA_Map Georeferencing Challenge_Validation Answer Key/'
IMG_FILE_EXT = 'tif'
CLUE_FILEN_SUFFIX = '_clue'

Image.MAX_IMAGE_PIXELS = 400000000
IMG_CACHE = 'temp/images/'
OCR_CACHE = 'temp/ocr/'
GEOCODE_CACHE = 'temp/geocode/'
os.makedirs(IMG_CACHE, exist_ok=True)
os.makedirs(OCR_CACHE, exist_ok=True)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/phorne/google-vision-lara.json'

def create_input(raster_id:str, image_path:str, query_path:str):
    input = PipelineInput()
    input.image = Image.open(image_path)
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_geofence()
    input.params['lon_minmax'] = lon_minmax
    input.params['lat_minmax'] = lat_minmax
    input.params['lon_sign_factor'] = lon_sign_factor


    query_pts = parse_query_file(query_path, input.image.size)
    input.params['query_pts'] = query_pts
    
    return input

def process_folder(image_folder:str):
    # get the pipelines
    pipelines = create_geo_referencing_pipelines()

    img_path_tmp = os.path.join(image_folder, '*.' + IMG_FILE_EXT)
    image_filenames = glob.glob(img_path_tmp)
    image_filenames.sort()

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
        print(f'processing {image_path}')
        image_base_filename = os.path.basename(image_path)
        image_base_filename = os.path.splitext(image_base_filename)[0]
        raster_id = image_base_filename

        query_path = os.path.join(QUERY_PATH_IN, image_base_filename + '.csv')

        input = create_input(raster_id, image_path, query_path)
        if input.params['query_pts'] is None or len(input.params['query_pts']) == 0:
            continue

        for pipeline in pipelines:
            output = pipeline.run(input)
            results[pipeline.id].append(output['geo'])
            results_summary[pipeline.id].append(output['summary'])
            results_levers[pipeline.id].append(output['levers'])
            results_gcps[pipeline.id].append(output['gcps'])
            results_integration[pipeline.id].append(output['schema'])
            print(f'done pipeline {pipeline.id}\n\n')
    
        for p in pipelines:
            writer_csv.output(results[p.id], {'path': f'output/test-{p.id}.csv'})
            writer_csv.output(results_summary[p.id], {'path': f'output/test_summary-{p.id}.csv'})
            writer_json.output(results_levers[p.id], {'path': f'output/test_levers-{p.id}.json'})
            writer_json.output(results_gcps[p.id], {'path': f'output/test_gcps-{p.id}.json'})
            writer_json.output(results_integration[p.id], {'path': f'output/test_schema-{p.id}.json'})
    
    for p in pipelines:
        writer_csv.output(results[p.id], {'path': f'output/test-{p.id}.csv'})
        writer_csv.output(results_summary[p.id], {'path': f'output/test_summary-{p.id}.csv'})
        writer_json.output(results_levers[p.id], {'path': f'output/test_levers-{p.id}.json'})
        writer_json.output(results_gcps[p.id], {'path': f'output/test_gcps-{p.id}.json'})
        writer_json.output(results_integration[p.id], {'path': f'output/test_schema-{p.id}.json'})


def get_geofence(lon_limits=(-66.0, -180.0), lat_limits=(24.0, 73.0), use_abs=True):
    lon_minmax = lon_limits
    lat_minmax = lat_limits

    lon_sign_factor = 1.0

    if use_abs: # use abs of lat/lon geo-fence? (since parsed OCR values don't usually include sign)
        if lon_minmax[0] < 0.0:
             lon_sign_factor = -1.0     # to account for -ve longitude values being forced to abs
                                        # (used when finalizing lat/lon results for query points) 
        lon_minmax = [abs(x) for x in lon_minmax]
        lat_minmax = [abs(x) for x in lat_minmax]

        lon_minmax = [min(lon_minmax), max(lon_minmax)]
        lat_minmax = [min(lat_minmax), max(lat_minmax)]


    return (lon_minmax, lat_minmax, lon_sign_factor)

def parse_query_file(csv_query_file, image_size=None):

    '''
    Expected schema is of the form:
    raster_ID,row,col,NAD83_x,NAD83_y
    GEO_0004,8250,12796,-105.72065081057087,43.40255034572461
    ...
    Note: NAD83* columns may not be present
    row (y) and col (x) = pixel coordinates to query
    NAD83* = (if present) are ground truth answers (lon and lat) for the query x,y pt
    '''

    first_line = True
    x_idx = 2
    y_idx = 1
    lon_idx = 3
    lat_idx = 4
    query_pts = []
    try:
        with open(csv_query_file) as f_in:
            for line in f_in:
                if line.startswith('raster_') or first_line:
                    first_line = False
                    continue    # header line, skip

                rec = line.split(',')
                if len(rec) < 3:
                    continue
                raster_id = rec[0]
                x = int(rec[x_idx])
                y = int(rec[y_idx])
                if image_size is not None:
                    # sanity check that query points are not > image dimensions!
                    if x > image_size[0] or y > image_size[1]:
                        err_msg = 'Query point {}, {} is outside image dimensions'.format(x,y)
                        raise IOError(err_msg)
                lonlat_gt = None
                if len(rec) >= 5:
                    lon = float(rec[lon_idx])
                    lat = float(rec[lat_idx])
                    if lon != 0 and lat != 0:
                        lonlat_gt = (lon, lat)
                query_pts.append(QueryPoint(raster_id, (x,y), lonlat_gt) )
                
    except Exception as e:
        print('EXCEPTION parsing query file: {}'.format(csv_query_file))
        print(e)

    return query_pts