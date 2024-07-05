from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from schema.cdr_schemas.prospectivity_input import StackMetaData


class Accelerator(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class NeuralNetTrainConfig(BaseModel):
    min_epochs: int  # prevents early stopping
    max_epochs: int

    accelerator: Accelerator

    # mixed precision for extra speed-up
    precision: int

    # perform a validation loop twice every training epoch
    val_check_interval: float

    # set True to to ensure deterministic results
    # makes training slower but gives more reproducibility than just setting seeds
    deterministic: bool


class NeighborhoodFunction(str, Enum):
    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"


class SOMType(str, Enum):
    TOROID = "toroid"
    SHEET = "sheet"


class NeighborhoodDecay(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class LearningRateDecay(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class SOMInitialization(str, Enum):
    RANDOM = "random"
    PCA = "pca"


class SOMGrid(str, Enum):
    HEXAGONAL = "hexagonal"
    RECTANGULAR = "rectangular"


class SOMTrainConfig(BaseModel):
    dimensions_x: int
    dimensions_y: int
    num_epochs: int
    num_initializations: int
    neighborhood_function: NeighborhoodFunction
    som_type: SOMType
    neighborhood_decay: NeighborhoodDecay
    learning_rate_decay: LearningRateDecay
    initial_learning_rate: float
    final_learning_rate: float
    som_initialization: SOMInitialization
    som_grid: SOMGrid


class NeuralNetModel(BaseModel):
    train_config: NeuralNetTrainConfig
    pass


class SOMModel(BaseModel):
    train_config: SOMTrainConfig
    pass


class CMAModel(BaseModel):
    title: Optional[str] = Field(
        ...,
        description="""
            Title of the model.
        """,
    )
    date: Optional[int] = Field(
        ...,
        description="""
            Date that the model was made. i.e. 2012
        """,
    )
    authors: Optional[List[str]] = Field(
        ...,
        description="""
            Creators of the model
        """,
    )
    organization: Optional[str] = Field(
        ...,
        description="""
            Organization that created the model
        """,
    )
    cma_model_type: Union[NeuralNetModel, SOMModel]

    training_data: StackMetaData
