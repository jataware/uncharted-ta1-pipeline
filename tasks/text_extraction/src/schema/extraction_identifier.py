
from pydantic import BaseModel
from typing import List, Tuple


class ExtractionIdentifier(BaseModel):
    '''
    Extraction Identifier object (from TA1 schema) 
    '''

    model: str              # model name
    field: str = None       # Field name of the model
    id: str = None          # ID of the extracted feature



        