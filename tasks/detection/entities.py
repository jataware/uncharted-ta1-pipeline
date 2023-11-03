from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from pydantic import BaseModel, validator, model_serializer
import torch
from typing import Optional, List, Dict, Union, Any


## Data Objects

class MapPointLabel(BaseModel):
    """
    Represents a label on a map image.
    Class ID should correspond to the ID encoded in the underlying model.
    """
    classifier_name: str
    classifier_version: int
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    directionality: Optional[Dict] = None

    def serialize(self):
        return dict(self)

class MapImage(BaseModel):
    """
    Represents a map image.
    """
    path: str
    labels: Optional[List[MapPointLabel]] = None
    map_bounds: Optional[List[int]] = None # [x1, y1, h, w] location of map. TODO: Accept polygonal seg mask.
    point_legend_bounds: Optional[List[int]] = None # [x1, y1, h, w] location of point legend.
    polygon_legend_bounds: Optional[List[int]] = None # [x1, y1, h, w] location of polygon legend.

    _cached_image = None

    @property
    def size(self):
        # Make use of the PIL Image.open lazy image loading to avoid loading the image prematurely.
        return self.image.size

    @property
    def image(self):
        if self._cached_image:
            img = self._cached_image
        else:
            img = Image.open(self.path)
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError("Image cannot have 0 height or width")
            self._cached_image = img
        # TODO: Use polygonal segmask stored in self.map_bounds to filter the image and crop out the non-map regions.
        return img


    @classmethod
    def load(cls, path: str,
             labels: Optional[List[MapPointLabel]] = None,
             map_bounds: Optional[List[int]] = None):
        image = Image.open(path)
        return cls(path=path,
                   image=image,
                   labels=labels,
                   map_bounds=map_bounds)

    def _serialize_labels(self):
        if self.labels:
            labels = [label.serialize() for label in self.labels]
        else:
            labels = None
        return labels

    @model_serializer
    def serialize(self):

        return {
            'map_path': self.path,
            'map_bounds': self.map_bounds,
            'point_legend_bounds': self.point_legend_bounds,
            'labels': self._serialize_labels(),
        }


class MapTile(BaseModel):
    """
    Represents a tile of a map image in (x, y, width, height) format.
    x and y are coordinates on the original map image.

    Image tensors are assumed to be in Torchvision format (C, H, W). These are automatically converted to PIL Images.
    """
    x_offset: int  # x offset of the tile in the original image.
    y_offset: int  # y offset of the tile in the original image.
    width: int
    height: int
    image: Any # torch.Tensor or PIL.Image
    map_path: str  # Path to the original map image.
    predictions: Union[Any, None] = None

    @validator("image", pre=True, always=True)
    def validate_image(cls, value):
        if isinstance(value, torch.Tensor):
            value = value.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            value = value.numpy()
            if value.dtype == np.float32:
                value = (value * 255).astype(np.uint8)
            value = Image.fromarray(value)
        if not isinstance(value, Image.Image):
            raise TypeError(f"Expected PIL or torch.Tensor Image, got {type(value)}")
        if value.size[0] == 0 or value.size[1] == 0:
            raise ValueError("Image cannot have 0 height or width")

        return value

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def load(cls, path: str, predictions: Optional[List[MapPointLabel]] = None):
        image = Image.open(path)
        return cls(path=path,
                   image=image,
                   predictions=predictions)

    def img_to_torchvision_tensor(self):
        return torch.tensor(self.image).float().permute(2, 0, 1) # Convert from (H, W, C) to (C, H, W)


## Pipeline Objects

class Task(ABC):
    """
    Abstract class for preprocessing maps before prediction.
    """

    @abstractmethod
    def process(self, *args, **kwargs) -> Union[MapImage, MapTile]:
        pass

    @property
    @abstractmethod
    def input_type(self):
        pass

    @property
    @abstractmethod
    def output_type(self):
        pass


class Pipeline():

    def __init__(self,
                 steps: List[Task]):
        self.steps = steps
        assert len(self.steps) > 0, "Pipeline must have at least one step."
        self._validate_steps()

    def _validate_steps(self):
        """
        Validate that the output of one step matches the input of the next.
        """
        for i in range(len(self.steps) - 1):
            if len(self.steps) == 1:
                print("Warning: Pipeline has only one step. This is not recommended.")
                break
            output_type = self.steps[i].output_type
            input_type = self.steps[i + 1].input_type
            assert output_type == input_type, f"Step {i} output type {output_type} does not match step {i + 1} input type {input_type}"

    def process(self, image: Union[MapImage, MapTile]) -> Union[MapImage, MapTile]:
        """
        Process an image through the pipeline.
        """
        for step in self.steps:
            image = step.process(image)
        return image
