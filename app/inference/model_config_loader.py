from pathlib import Path

import yaml
from pydantic import ValidationError

from app.domain.model import ModelCode, ModelSpec, ServingModelConfig


class ModelConfigError(RuntimeError):
    """모델 설정을 읽거나 검증할 수 없을 때 발생하는 오류."""


class ModelConfigLoader:
    def __init__(
        self,
        config_path: str | Path = "configs/models.yaml",
        project_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)

        if project_root is None:
            self.project_root = self.config_path.parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()

    def load(self) -> ServingModelConfig:
        if not self.config_path.is_file():
            raise ModelConfigError(
                f"Model configuration file was not found: {self.config_path}"
            )

        try:
            with self.config_path.open(
                mode="r",
                encoding="utf-8-sig",
            ) as config_file:
                raw_config = yaml.safe_load(config_file)

        except OSError as error:
            raise ModelConfigError(
                f"Failed to read model configuration: {error}"
            ) from error

        except yaml.YAMLError as error:
            raise ModelConfigError(
                f"Invalid YAML syntax: {error}"
            ) from error

        if not isinstance(raw_config, dict):
            raise ModelConfigError(
                "Model configuration root must be a YAML object."
            )

        try:
            return ServingModelConfig.model_validate(raw_config)

        except ValidationError as error:
            raise ModelConfigError(
                f"Model configuration validation failed:\n{error}"
            ) from error

    def resolve_model_path(self, model: ModelSpec) -> Path:
        configured_path = Path(model.file_path)

        if configured_path.is_absolute():
            return configured_path.resolve()

        return (self.project_root / configured_path).resolve()

    def validate_model_files(
        self,
        config: ServingModelConfig,
    ) -> dict[ModelCode, Path]:
        resolved_paths: dict[ModelCode, Path] = {}

        for model in config.get_enabled_models():
            model_path = self.resolve_model_path(model)

            if not model_path.is_file():
                raise ModelConfigError(
                    "Model file was not found for "
                    f"{model.model_code.value}: {model_path}"
                )

            resolved_paths[model.model_code] = model_path

        return resolved_paths

