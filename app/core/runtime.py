from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app.domain.model import ModelCode, ServingModelConfig


class RuntimeStatus(StrEnum):
    STARTING = "starting"
    CONFIG_VALIDATED = "config_validated"
    READY = "ready"
    NOT_READY = "not_ready"


@dataclass(slots=True)
class ApplicationRuntimeState:
    status: RuntimeStatus = RuntimeStatus.STARTING
    model_config: ServingModelConfig | None = None
    model_paths: dict[ModelCode, Path] = field(default_factory=dict)
    startup_error: str | None = None
    configuration_validated: bool = False

    @property
    def is_ready(self) -> bool:
        return self.status is RuntimeStatus.READY

    @property
    def is_config_validated(self) -> bool:
        return self.configuration_validated

    @property
    def configured_model_count(self) -> int:
        if self.model_config is None:
            return 0

        return len(self.model_config.get_enabled_models())

    @property
    def validated_model_count(self) -> int:
        return len(self.model_paths)

    def begin_startup(self) -> None:
        self.status = RuntimeStatus.STARTING
        self.model_config = None
        self.model_paths.clear()
        self.startup_error = None
        self.configuration_validated = False

    def set_model_config(
        self,
        model_config: ServingModelConfig,
    ) -> None:
        self.model_config = model_config

    def set_model_paths(
        self,
        model_paths: dict[ModelCode, Path],
    ) -> None:
        self.model_paths = dict(model_paths)

    def mark_config_validated(self) -> None:
        self.status = RuntimeStatus.CONFIG_VALIDATED
        self.startup_error = None
        self.configuration_validated = True

    def mark_ready(self) -> None:
        self.status = RuntimeStatus.READY
        self.startup_error = None
        self.configuration_validated = True

    def mark_not_ready(
        self,
        error: Exception | str,
    ) -> None:
        self.status = RuntimeStatus.NOT_READY
        self.startup_error = str(error)