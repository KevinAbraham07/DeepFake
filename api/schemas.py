from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """
    Standardized JSON response for the /predict/image endpoint.
    """
    filename: str = Field(..., description="The name of the uploaded file")
    prediction: str = Field(..., description="Either 'Fake' or 'Real'")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    processing_time_ms: float = Field(..., description="Time taken to run inference in milliseconds")
