from pydantic import BaseModel, Field
from typing import List, Optional

class ModelBreakdownItem(BaseModel):
    model_name: str = Field(..., description="Name / Description of the model")
    is_fake: bool = Field(..., description="Prediction of this individual model")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")

class PredictionResponse(BaseModel):
    filename: str = Field(..., description="The name of the uploaded file")
    is_fake: bool = Field(..., description="Overall combined prediction")
    confidence: float = Field(..., description="Overall combined confidence score")
    processing_time: float = Field(..., description="Time taken in seconds")
    model_name: str = Field(default="Ensemble Engine (Combined)", description="Engine name")
    model_breakdown: List[ModelBreakdownItem] = Field(default_factory=list, description="Per-model prediction breakdown")
