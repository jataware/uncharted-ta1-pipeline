
from random import randint

from tasks.common.task import (Task, TaskInput, TaskResult)
from tasks.geo_referencing.georeference import QueryPoint

class CreateGroundControlPoints(Task):

    def __init__(self, task_id:str):
        super().__init__(task_id)
    
    def run(self, input:TaskInput) -> TaskResult:
        print(f'running ground control point creation at task index {input.task_index} with id {self._task_id}')
        # check if query points already defined
        query_pts = input.request['query_pts']
        if query_pts and len(query_pts) > 0:
            print('ground control points already exist')
            return self._create_result(input)

        # no query points exist, so create them
        query_pts = self._create_query_points(input)
        print(f'created {len(query_pts)} ground control points')

        # add them to the output
        result = self._create_result(input)
        result.output['query_pts'] = query_pts

        return result
    
    def _create_query_points(self, input:TaskInput) -> list[QueryPoint]:
        # create 10 ground control points roughly around the middle of the ROI (or failing that the middle of the image)
        min_x = min_y = max_x = max_y = 0
        roi = input.get_data('roi')
        if roi and len(roi) > 0:
            roi_x = list(map(lambda x: x[0], roi))
            roi_y = list(map(lambda x: x[1], roi))

            max_x, max_y = max(roi_x), max(roi_y)
            min_x, min_y = min(roi_x), min(roi_y)
        else:
            max_x, max_y = input.image.size
        
        coords = self._create_random_coordinates(min_x, min_y, max_x, max_y)
        return [QueryPoint(input.raster_id, (c[0],c[1]), None, properties={'label': 'random'}) for c in coords]
    
    def _create_random_coordinates(self, min_x:float, min_y:float, max_x:float, max_y:float, n:int=10, buffer:float=0.25) -> list[tuple[int, int]]:
        # randomize x & y coordinates fitting between boundaries
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        min_x_buf = min_x + range_x * buffer
        max_x_buf = max_x - range_x * buffer
        min_y_buf = min_y + range_y * buffer
        max_y_buf = max_y - range_y * buffer

        return [(randint(int(min_x_buf), int(max_x_buf)), randint(int(min_y_buf), int(max_y_buf))) for _ in range(n)]