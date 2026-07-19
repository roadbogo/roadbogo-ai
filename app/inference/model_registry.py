from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.domain.model import ModelCode, ModelSpec, ServingModelConfig
from app.inference.model_loader import InferenceModel, LoadedModel


class ModelRegistryError(RuntimeError):
    """?? ?????? ????? ??? ? ?? ? ???? ??."""


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    spec: ModelSpec
    file_path: Path
    model: InferenceModel | None = None
    device: str | None = None
    half_precision: bool | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[ModelCode, RegisteredModel] = {}

    @property
    def is_configured(self) -> bool:
        return bool(self._models)

    @property
    def is_loaded(self) -> bool:
        return (
            self.is_configured
            and all(
                registered.is_loaded
                for registered in self._models.values()
            )
        )

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def loaded_model_count(self) -> int:
        return sum(
            registered.is_loaded
            for registered in self._models.values()
        )

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
            missing = sorted(
                code.value
                for code in missing_codes
            )
            unexpected = sorted(
                code.value
                for code in unexpected_codes
            )

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

    def attach_loaded_models(
        self,
        loaded_models: Mapping[ModelCode, LoadedModel],
    ) -> None:
        if not self.is_configured:
            raise ModelRegistryError(
                "Model registry is not configured."
            )

        registered_codes = set(self._models)
        loaded_codes = set(loaded_models)

        missing_codes = registered_codes - loaded_codes
        unexpected_codes = loaded_codes - registered_codes

        if missing_codes or unexpected_codes:
            missing = sorted(
                code.value
                for code in missing_codes
            )
            unexpected = sorted(
                code.value
                for code in unexpected_codes
            )

            raise ModelRegistryError(
                "Loaded models do not match registered models. "
                f"missing={missing}, unexpected={unexpected}"
            )

        attached_models: dict[
            ModelCode,
            RegisteredModel,
        ] = {}

        for model_code, registered in self._models.items():
            loaded = loaded_models[model_code]

            if loaded.spec != registered.spec:
                raise ModelRegistryError(
                    "Loaded model specification does not match "
                    f"registry: {model_code.value}"
                )

            if loaded.file_path != registered.file_path:
                raise ModelRegistryError(
                    "Loaded model path does not match registry: "
                    f"{model_code.value}"
                )

            attached_models[model_code] = RegisteredModel(
                spec=registered.spec,
                file_path=registered.file_path,
                model=loaded.model,
                device=loaded.device,
                half_precision=loaded.half_precision,
            )

        self._models = attached_models

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
