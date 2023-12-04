from PIL.Image import Image as PILImage

from tasks.geo_referencing.task import Task, TaskInput, TaskResult

from typing import Optional


class PipelineInput:
    image: Optional[PILImage] = None
    raster_id: str = ""
    params = {}

    def __init__(self):
        self.image = None
        self.raster_id = ""
        self.params = {}


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


class OutputCreator:
    id = ""

    def __init__(self, id):
        self.id = id

    def create_output(self, pipeline_result: PipelineResult) -> Output:
        return Output(pipeline_result.pipeline_id, pipeline_result.pipeline_name)


class Pipeline:
    id = ""
    name = ""
    _tasks: list[Task] = []
    _output: list[OutputCreator] = []

    def __init__(
        self, id: str, name: str, output: list[OutputCreator], tasks: list[Task] = []
    ) -> None:
        self._tasks = tasks
        self._output = output
        self.id = id
        self.name = name

    def run(self, input: PipelineInput) -> dict[str, Output]:
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
                    f"EXCEPTION executing pipeline at step {task_input.task_index} for raster {input.raster_id}"
                )
                print(e)

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

    def _produce_output(self, pipeline_result: PipelineResult) -> dict[str, Output]:
        outputs = {}
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
