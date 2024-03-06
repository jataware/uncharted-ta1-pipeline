import copy, json, logging, os
from PIL.Image import Image as PILImage

from typing import Callable, List, Any, Dict

logger = logging.getLogger(__name__)


class TaskParameter:
    task_id: str = ""
    task_index: int = -1
    key: str = ""
    category: str = ""
    description: str = ""
    values: Dict[str, Any] = {}

    def __init__(
        self,
        task_id: str,
        task_index: int,
        key: str,
        category: str,
        values: Dict[str, Any],
        description: str = "",
    ):
        self.task_id = task_id
        self.task_index = task_index
        self.key = key
        self.category = category
        self.values = values
        self.description = description


class TaskInput:
    image: PILImage
    task_index: int
    raster_id: str = ""
    request: Dict[Any, Any] = {}
    data: Dict[Any, Any] = {}
    params_used: List[TaskParameter] = []

    def __init__(self, task_index: int):
        self.data = {}
        self.raster_id = ""
        self.request = {}
        self.params_used = []
        self.task_index = task_index

    def get_data(self, key: str, default_value: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        return default_value

    def parse_data(self, key: str, parser: Callable, default_value: Any = None) -> Any:
        if key in self.data:
            return parser(self.data[key])
        return default_value

    def get_request_info(self, key: str, default_value: Any = None) -> Any:
        if key in self.request:
            return copy.deepcopy(self.request[key])
        return default_value

    def add_param(
        self,
        task_id: str,
        key: str,
        category: str,
        values: Dict[str, Any],
        description: str = "",
    ):
        self.params_used.append(
            TaskParameter(task_id, self.task_index, key, category, values, description)
        )


class TaskResult:
    def __init__(
        self,
        task_id: str = "",
        output: Dict[Any, Any] = {},
        parameters: List[TaskParameter] = [],
    ):
        self.task_id = task_id
        self.output = {} if output == {} else output
        self.parameters = [] if parameters == [] else parameters

    def add_output(self, key: str, data: Dict[Any, Any]):
        self.output[key] = data


class Task:
    _task_id = ""

    def __init__(self, task_id: str, cache_dir: str = ""):
        self._task_id = task_id
        self._cache_dir = cache_dir

        if self._cache_dir:
            logger.info(
                f"Initializing task {self._task_id} with cache dir {self._cache_dir}"
            )
            self._init_cache()
        else:
            logger.info(f"Initializing task {self._task_id} with no local data cache")

    def run(self, input: TaskInput) -> TaskResult:
        logger.info(f"Running task {self._task_id}")

        result = TaskResult()
        result.task_id = self._task_id
        return result

    def get_task_id(self) -> str:
        return self._task_id

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
        values: Dict[str, Any],
        description: str = "",
    ):
        input.add_param(self._task_id, key, category, values, description)

    def _init_cache(self):
        """
        Create local cache dir, if it doesn't exist
        """
        if self._cache_dir and not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

    def _get_cache_doc_path(self, doc_key: str) -> str:
        """
        Generate the full local path for cached json result
        """
        return os.path.join(self._cache_dir, f"{doc_key}.json")

    def fetch_cached_result(self, doc_key: str) -> Any:
        """
        Check if task result is available in the local cache
        """
        if not self._cache_dir:
            return None
        doc_path = self._get_cache_doc_path(doc_key)
        if os.path.isfile(doc_path):
            with open(doc_path, "rb") as f:
                return json.load(f)
        else:
            return None

    def write_result_to_cache(self, json_model: Dict[Any, Any], doc_key: str):
        """
        Write task result to local cache
        """
        if not self._cache_dir:
            return
        doc_path = self._get_cache_doc_path(doc_key)
        with open(doc_path, "w") as f:
            json.dump(json_model, f)
