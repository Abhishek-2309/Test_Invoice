from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ExtractionResponse(BaseModel):
    filename: str
    elapsed_seconds: float
    fields: Dict[str, Any]
    model_info: Dict[str, Any]
    ocr_markdown: Optional[str] = None

class BatchExtractionResponse(BaseModel):
    total: int
    errors: int
    results_raw: List[str] = Field(default_factory=list)
