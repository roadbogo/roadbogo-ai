from pydantic import BaseModel, ConfigDict, Field

from app.domain.model import ModelCode


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0)
    raw_name: str
    class_code: str
    incident_target: bool
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox


class ModelDetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_code: ModelCode
    display_name: str
    detection_count: int = Field(ge=0)
    detections: list[Detection]


class ImageInferenceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)
    model_count: int = Field(ge=0)
    total_detection_count: int = Field(ge=0)
    incident_detection_count: int = Field(ge=0)
    model_results: list[ModelDetectionResult]
