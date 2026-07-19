from dataclasses import dataclass
from pathlib import Path

from app.domain.model import ModelCode, ModelSpec, ServingModelConfig


class ModelRegistryError(RuntimeError):
    """모델 레지스트리를 구성하거나 조회할 수 없을 때 발생하는 오류."""


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    spec: ModelSpec
    file_path: Path


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[ModelCode, RegisteredModel] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self._models)

    @property
    def model_count(self) -> int:
        return len(self._models)

    def register(
        self,
        config: ServingModelConfig,
        model_paths: dict[ModelCode, Path],
    ) -> None:
        enabled_models = config.get_enabled_models()
        enabled_codes = {
            model.model_code
            for model in enabled_models
        }
        path_codes = set(model_paths)

        missing_codes = enabled_codes - path_codes
        unexpected_codes = path_codes - enabled_codes

        if missing_codes or unexpected_codes:
            missing = sorted(code.value for code in missing_codes)
            unexpected = sorted(code.value for code in unexpected_codes)

            raise ModelRegistryError(
                "Model registry paths do not match enabled models. "
                f"missing={missing}, unexpected={unexpected}"
            )

        self._models = {
            model.model_code: RegisteredModel(
                spec=model,
                file_path=model_paths[model.model_code],
            )
            for model in enabled_models
        }

    def clear(self) -> None:
        self._models.clear()

    def get(self, model_code: ModelCode) -> RegisteredModel:
        try:
            return self._models[model_code]
        except KeyError as error:
            raise ModelRegistryError(
                f"Model is not registered: {model_code.value}"
            ) from error

    def list_models(self) -> tuple[RegisteredModel, ...]:
        return tuple(self._models.values())