from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    route_id: str = Field(..., examples=["sample-route"])
    features: dict[str, float] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    route_id: str
    score: float
    label: str
