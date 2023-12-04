
import copy

from PIL.Image import Image as PILImage
from typing import Optional, List, Any, Dict


class TaskParameter:
    task_id: str = ""
    task_index: int = -1
    key: str = ""
    category: str = ""
    description: str = ""
    value: Any = None

    def __init__(
        self,
        task_id: str,
        task_index: int,
        key: str,
        category: str,
        value: Any,
        description: str = "",
    ):
        self.task_id = task_id
        self.task_index = task_index
        self.key = key
        self.category = category
        self.value = value
        self.description = description


class TaskInput:
    image: Optional[PILImage] = None
    task_index: int
    raster_id: str = ""
    request: Dict[Any, Any] = {}
    data: Dict[Any, Any] = {}
    params_used: List[TaskParameter] = []

    def __init__(self, task_index: int):
        self.data = {}
        self.image = None
        self.raster_id = ""
        self.request = {}
        self.params_used = []
        self.task_index = task_index

    def get_data(self, key: str, default_value:Any=None) -> Any:
        if key in self.data:
            return self.data[key]
        return default_value
    
    def get_request_info(self, key:str, default_value:Any=None) -> Any:
        if key in self.request:
            return copy.deepcopy(self.request[key])
        return default_value

    def add_param(
        self, task_id: str, key: str, category: str, value: Any, description: str = ""
    ):
        self.params_used.append(
            TaskParameter(task_id, self.task_index, key, category, value, description)
        )

class TaskResult:
    task_id = ""
    output: Dict[Any, Any] = {}
    parameters: List[TaskParameter] = []

    def __init__(self):
        self.task_id = ""
        self.output = {}
        self.parameters = []


class Task:
    _task_id = ""

    def __init__(self, task_id):
        self._task_id = task_id

    def run(self, input: TaskInput) -> TaskResult:
        print(f"running task {self._task_id}")

        result = TaskResult()
        result.task_id = self._task_id
        return result

    def _create_result(self, input: TaskInput) -> TaskResult:
        result = TaskResult()
        result.task_id = self._task_id
        result.parameters = input.params_used.copy()

        return result

    def _add_param(
        self,
        input: TaskInput,
        key: str,
        category: str,
        value: Any,
        description: str = "",
    ):
        input.add_param(self._task_id, key, category, value, description)