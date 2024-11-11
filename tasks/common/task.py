import copy, json, logging, os
from PIL.Image import Image as PILImage

from typing import Callable, List, Any, Dict, Optional

from pydantic import BaseModel

from tasks.common.io import (
    JSONFileReader,
    JSONFileWriter,
    Mode,
    append_to_cache_location,
    bucket_exists,
    get_file_source,
)

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


class HaltPipeline(TaskResult):
    """
    HaltPipeline is a task result which flags a request to immediately halt pipeline execution.  This
    is useful for stopping a pipeline when some necessary condition is not met, such as no map being
    present in a supplied image.

    Attributes:
        task_id (str): The unique identifier for the task.
        reason (str): The reason for halting the pipeline.
        output (dict): An empty dictionary to store output data.
        parameters (list): An empty list to store parameters.

    Methods:
        __init__(task_id: str, reason: str): Initializes the HaltPipeline instance with the given task ID and reason.
    """

    def __init__(self, task_id: str, reason: str):
        super().__init__(task_id)
        self.output = {}
        self.parameters = []
        self.reason = reason


class Task:
    _task_id = ""

    def __init__(self, task_id: str, cache_location: str = ""):
        self._task_id = task_id
        self._cache_location = cache_location

        self._json_file_reader = JSONFileReader()
        self._json_file_writer = JSONFileWriter()

        if self._cache_location:
            logger.info(
                f"Initializing task {self._task_id} with cache location {self._cache_location}"
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
        If working with the file system, create local cache dir if it doesn't exist.  If working with S3,
        ensure the bucket exists and is accessible.
        """
        cache_mode = get_file_source(self._cache_location)
        if cache_mode == Mode.FILE:
            if not os.path.exists(self._cache_location):
                os.makedirs(self._cache_location)
        elif cache_mode == Mode.S3_URI or Mode.URL:
            if not bucket_exists(self._cache_location):
                raise Exception(
                    f"S3 cache bucket {self._cache_location} does not exist"
                )
        else:
            raise Exception(f"Invalid cache location {self._cache_location}")

    def _get_cache_doc_path(self, doc_key: str) -> str:
        """
        Generate the full local path for cached json result
        """
        return append_to_cache_location(self._cache_location, f"{doc_key}.json")

    def fetch_cached_result(self, doc_key: str) -> Optional[Dict[Any, Any]]:
        """
        Check if task result is available in the local cache
        """
        if not self._cache_location:
            return None
        try:
            return self._json_file_reader.process(self._get_cache_doc_path(doc_key))
        except Exception:
            return None

    def write_result_to_cache(self, json_model: BaseModel, doc_key: str):
        """
        Write task result to local cache
        """
        if not self._cache_location:
            return
        doc_path = self._get_cache_doc_path(doc_key)
        self._json_file_writer.process(doc_path, json_model)


class EvaluateHalt(Task):
    """
    EvaluateHalt is a Task that evaluates a supplied halt condition.

    Attributes:
        _eval_halt (Callable[[TaskInput], bool]): A callable that takes a TaskInput and returns a boolean indicating whether to halt the pipeline.
    """

    def __init__(
        self,
        task_id: str,
        eval_halt: Callable[[TaskInput], bool],
        cache_location: str = "",
    ):
        """
        Initialize a new task instance.
        Args:
            task_id (str): The unique identifier for the task.
            eval_halt (Callable[[TaskInput], bool]): A callable that determines
                whether the task should halt based on the given TaskInput.
            cache_location (str, optional): The location to cache task data.
                Defaults to an empty string.
        """

        super().__init__(task_id, cache_location)
        self._eval_halt = eval_halt

    def run(self, input: TaskInput) -> TaskResult:
        """
        Executes the task with the given input and returns the result.
        Args:
            input (TaskInput): The input data required to run the task.
        Returns:
            TaskResult: The result of the task execution. If the halt condition is met,
                        returns a HaltPipeline instance indicating the task should be halted.
        """
        logger.info("Running EvaluateHalt task")
        if self._eval_halt(input) is True:
            return HaltPipeline(self._task_id, f"Halt condition met - {self._task_id}")
        return TaskResult(self._task_id)
