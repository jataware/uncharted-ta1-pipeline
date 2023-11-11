
from pydantic import BaseModel
from typing import List, Tuple
from schema.extraction_identifier import ExtractionIdentifier


class PageExtraction(BaseModel):
    '''
    Page Extraction object (from TA1 schema) 
    '''

    name: str                # label (class) of extracted object
    bounds: List[Tuple]      # segmentation polygon, list of xy co-ords (in pixel units)
    
    confidence: float = None              # [Optional] confidence score for this extraction
    ocr_text: str = None                  # [Optional] any OCR text associated with this extraction
    color_estimation: str = None          # [Optional] estimated color associated with this extraction (in hex string format)
    model: ExtractionIdentifier = None    # [Optional] Model information associated with this extraction

        