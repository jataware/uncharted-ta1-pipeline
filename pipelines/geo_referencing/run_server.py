
import logging
import os

from flask import Flask, request, Response
from PIL import Image

from pipelines.geo_referencing.factory import create_geo_referencing_pipeline
from pipelines.geo_referencing.output import (JSONWriter)
from pipelines.geo_referencing.pipeline import (PipelineInput)
from tasks.geo_referencing.georeference import QueryPoint

Image.MAX_IMAGE_PIXELS = 400000000


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/credentials.json'

app = Flask(__name__)

def load_image(image_path:str) -> Image:
    # assume local file system for now
    return Image.open(image_path)

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

def create_query_points(raster_id:str, points) -> list[QueryPoint]:
    return [QueryPoint(raster_id, (p['x'], p['y']), None) for p in points]

def create_input(raster_id:str, image_path:str, points) -> PipelineInput:
    input = PipelineInput()
    input.image = load_image(image_path)
    input.raster_id = raster_id

    lon_minmax, lat_minmax, lon_sign_factor = get_geofence()
    input.params['lon_minmax'] = lon_minmax
    input.params['lat_minmax'] = lat_minmax
    input.params['lon_sign_factor'] = lon_sign_factor


    query_pts = create_query_points(raster_id, points)
    input.params['query_pts'] = query_pts
    
    return input

def process_input(raster_id:str, image_path:str, points):
    # create the input for the pipeline
    input = create_input(raster_id, image_path, points)

    # create the pipeline
    pipeline = create_geo_referencing_pipeline()

    # run the pipeline
    outputs = pipeline.run(input)

    # create the output assuming schema output is part of the pipeline
    output_schema = outputs['schema']
    writer_json = JSONWriter()
    return writer_json.output([output_schema], {})

@app.route('/api/process_image', methods=['POST'])
def process_image():
    # get input values
    data = request.json
    raster_id = data['id']
    image_path = data['image_path']
    points = data['points']
    
    # process the request
    result = process_input(raster_id, image_path, points)
    return Response(result, status=200, mimetype="application/json")


@app.route("/healthcheck")
def health():
    """
    healthcheck
    """
    return ("healthy", 200)


def start_server():
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s %(levelname)s %(name)s\t: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('georef app')
    logger.info('*** Starting geo referencing app ***')
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    start_server()