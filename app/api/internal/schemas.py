from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from app.domain.model import ModelCode


class CaptureSource(StrEnum):
    LIVE = "LIVE"
    DEMO = "DEMO"
    UPLOAD = "UPLOAD"


class ContractExecutionMode(StrEnum):
    PARALLEL = "PARALLEL"
    SEQUENTIAL = "SEQUENTIAL"


class ContractModelCode(StrEnum):
    VEHICLE_DETECTOR = "VEHICLE_DETECTOR"
    DEBRIS_DETECTOR = "DEBRIS_DETECTOR"
    WILDLIFE_DETECTOR = "WILDLIFE_DETECTOR"


CONTRACT_TO_INTERNAL_MODEL_CODE = {
    ContractModelCode.VEHICLE_DETECTOR: (
        ModelCode.PROHIBITED_MOBILITY_DETECTOR
    ),
    ContractModelCode.DEBRIS_DETECTOR: (
        ModelCode.DEBRIS_DETECTOR
    ),
    ContractModelCode.WILDLIFE_DETECTOR: (
        ModelCode.WILDLIFE_DETECTOR
    ),
}


class VideoFrameRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    public_id: UUID
    cctv_public_id: UUID
    captured_at: datetime
    frame_sequence: int = Field(ge=0)
    original_width: int = Field(ge=1)
    original_height: int = Field(ge=1)
    capture_source: CaptureSource

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at_is_utc(
        cls,
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() != timedelta(0)
        ):
            raise ValueError(
                "captured_at must use ISO 8601 UTC."
            )

        return value


class InferenceInputRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    file_public_id: UUID
    download_url: HttpUrl
    mime_type: str

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(
        cls,
        value: str,
    ) -> str:
        allowed_types = {
            "image/jpeg",
            "image/png",
        }

        if value not in allowed_types:
            raise ValueError(
                "Only JPEG and PNG image MIME types "
                "are supported."
            )

        return value


class TrackingRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    tracking_session_key: str = Field(
        min_length=1,
        max_length=160,
    )


class InferenceExecutionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_codes: list[ContractModelCode] = Field(
        min_length=1,
    )
    execution_mode: ContractExecutionMode
    threshold_profile_code: str = Field(
        min_length=1,
        max_length=60,
    )

    @model_validator(mode="after")
    def validate_unique_model_codes(
        self,
    ) -> Self:
        if len(self.model_codes) != len(
            set(self.model_codes)
        ):
            raise ValueError(
                "model_codes must not contain duplicates."
            )

        return self

    def get_internal_model_codes(
        self,
    ) -> list[ModelCode]:
        return [
            CONTRACT_TO_INTERNAL_MODEL_CODE[model_code]
            for model_code in self.model_codes
        ]


class InternalInferenceRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    request_id: UUID
    inference_run_public_id: UUID
    video_frame: VideoFrameRequest
    input: InferenceInputRequest
    tracking: TrackingRequest
    execution: InferenceExecutionRequest
