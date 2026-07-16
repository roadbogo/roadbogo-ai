from enum import StrEnum
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExecutionMode(StrEnum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ModelCode(StrEnum):
    DEBRIS_DETECTOR = "DEBRIS_DETECTOR"
    PROHIBITED_MOBILITY_DETECTOR = "PROHIBITED_MOBILITY_DETECTOR"
    WILDLIFE_DETECTOR = "WILDLIFE_DETECTOR"


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExecutionMode
    device: str = Field(min_length=1)
    batch_size: int = Field(ge=1)
    half_precision: bool


class ModelClassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_name: str = Field(min_length=1)
    class_code: str = Field(
        min_length=1,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )
    incident_target: bool


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_code: ModelCode
    display_name: str = Field(min_length=1)
    file_path: str = Field(min_length=1)
    original_file_name: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    task: str = Field(min_length=1)

    input_size: int = Field(ge=32)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    enabled: bool

    classes: dict[int, ModelClassSpec]

    @field_validator("file_path")
    @classmethod
    def validate_model_file_extension(cls, value: str) -> str:
        if PurePosixPath(value).suffix.lower() != ".pt":
            raise ValueError("Model file_path must point to a .pt file.")

        return value

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        if not self.classes:
            raise ValueError("At least one model class must be configured.")

        class_ids = sorted(self.classes)
        expected_ids = list(range(len(class_ids)))

        if class_ids != expected_ids:
            raise ValueError(
                "Model class IDs must start at 0 and be continuous."
            )

        class_codes = [
            class_spec.class_code
            for class_spec in self.classes.values()
        ]

        if len(class_codes) != len(set(class_codes)):
            raise ValueError("Model class_code values must be unique.")

        return self


class ServingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1)
    execution: ExecutionSpec
    models: list[ModelSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_models(self) -> Self:
        model_codes = [model.model_code for model in self.models]

        if len(model_codes) != len(set(model_codes)):
            raise ValueError("model_code values must be unique.")

        return self

    def get_model(self, model_code: ModelCode | str) -> ModelSpec:
        target_code = ModelCode(model_code)

        for model in self.models:
            if model.model_code == target_code:
                return model

        raise KeyError(f"Model is not configured: {target_code.value}")

    def get_enabled_models(self) -> list[ModelSpec]:
        return [model for model in self.models if model.enabled]
