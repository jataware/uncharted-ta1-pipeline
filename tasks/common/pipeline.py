import io
from .task import Task, TaskInput, TaskResult
from typing import Optional, List, Dict, Any, Sequence
from PIL.Image import Image as PILImage
from pydantic import BaseModel
import traceback


class PipelineInput:
    image: Optional[PILImage] = None
    raster_id: str = ""
    params = {}

    def __init__(
        self,
        image: Optional[PILImage] = None,
        raster_id: str = "",
        params: Dict[str, Any] = {},
    ):
        self.image = image
        self.raster_id = raster_id
        self.params = {} if len(params) == 0 else params


class PipelineResult:
    pipeline_id = ""
    pipeline_name = ""
    image: Optional[PILImage] = None
    raster_id: str = ""
    data = {}
    tasks = {}
    result = {}
    params = []

    def __init__(self):
        self.pipeline_id = ""
        self.pipeline_name = ""
        self.image = None
        self.raster_id = ""
        self.data = {}
        self.tasks = {}
        self.result = {}
        self.params = []


class Output:
    pipeline_id = ""
    pipeline_name = ""

    def __init__(self, pipeline_id: str, pipeline_name: str):
        self.pipeline_id = pipeline_id
        self.pipeline_name = pipeline_name


class TabularOutput(Output):
    fields = []
    data = []

    def __init__(self, pipeline_id: str, pipeline_name: str):
        super().__init__(pipeline_id, pipeline_name)
        self.fields = []
        self.data = []


class ObjectOutput(Output):
    data = {}

    def __init__(self, pipeline_id: str, pipeline_name: str):
        super().__init__(pipeline_id, pipeline_name)
        self.data = {}


class ImageOutput(Output):
    data: PILImage

    def __init__(self, pipeline_id: str, pipeline_name: str, data: PILImage):
        super().__init__(pipeline_id, pipeline_name)
        self.data = data


class ImageDictOutput(Output):
    data: Dict[str, PILImage]

    def __init__(self, pipeline_id: str, pipeline_name: str, data: Dict[str, PILImage]):
        super().__init__(pipeline_id, pipeline_name)
        self.data = data


class ListOutput(Output):
    data = []

    def __init__(self, pipeline_id: str, pipeline_name: str):
        super().__init__(pipeline_id, pipeline_name)
        self.data = []


class BytesOutput(Output):
    data: io.BytesIO

    def __init__(self, pipeline_id: str, pipeline_name: str, data: io.BytesIO):
        super().__init__(pipeline_id, pipeline_name)
        self.data = data


class BaseModelOutput(Output):
    data: BaseModel

    def __init__(self, pipeline_id: str, pipeline_name: str, data: BaseModel):
        super().__init__(pipeline_id, pipeline_name)
        self.data = data


class BaseModelListOutput(Output):
    data: Sequence[BaseModel]

    def __init__(self, pipeline_id: str, pipeline_name: str, data: Sequence[BaseModel]):
        super().__init__(pipeline_id, pipeline_name)
        self.data = data


class OutputCreator:
    id = ""

    def __init__(self, id):
        self.id = id

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        raise NotImplementedError()


class Pipeline:
    id = ""
    name = ""
    _tasks: Sequence[Task] = []
    _output: Sequence[OutputCreator] = []

    def __init__(
        self,
        id: str,
        name: str,
        output: Sequence[OutputCreator],
        tasks: Sequence[Task] = [],
    ) -> None:
        self._tasks = tasks
        self._output = output
        self.id = id
        self.name = name

    def run(self, input: PipelineInput) -> Dict[str, Output]:
        pipeline_result = self._initialize_result(input)
        count = 0
        for t in self._tasks:
            task_input = self._create_task_input(count, input, pipeline_result)
            count = count + 1
            try:
                task_result = t.run(task_input)
                pipeline_result = self._merge_result(pipeline_result, task_result)
            except Exception as e:
                print(
                    f"EXCEPTION executing pipeline at step {t.get_task_id()} ({task_input.task_index}) for raster {input.raster_id}"
                )
                traceback.print_exc()

        return self._produce_output(pipeline_result)

    def _initialize_result(self, input: PipelineInput) -> PipelineResult:
        result = PipelineResult()
        result.data["image"] = input.image
        result.raster_id = input.raster_id
        result.image = input.image
        result.pipeline_id = self.id
        result.pipeline_name = self.name

        return result

    def _merge_result(
        self, pipeline_result: PipelineResult, task_result: TaskResult
    ) -> PipelineResult:
        # override result
        for k, v in task_result.output.items():
            pipeline_result.data[k] = v
        pipeline_result.tasks[task_result.task_id] = task_result

        # concatenate all params
        pipeline_result.params = pipeline_result.params + task_result.parameters

        return pipeline_result

    def _produce_output(self, pipeline_result: PipelineResult) -> Dict[str, Output]:
        outputs: Dict[str, Output] = {}
        for oc in self._output:
            outputs[oc.id] = oc.create_output(pipeline_result)
        return outputs

    def _create_task_input(
        self,
        task_index: int,
        pipeline_input: PipelineInput,
        pipeline_result: PipelineResult,
    ) -> TaskInput:
        task_input = TaskInput(task_index)
        task_input.image = pipeline_result.data["image"]
        task_input.raster_id = pipeline_input.raster_id
        task_input.data = {}
        for k, v in pipeline_result.data.items():
            task_input.data[k] = v
        for k, v in pipeline_input.params.items():
            task_input.request[k] = v

        return task_input
