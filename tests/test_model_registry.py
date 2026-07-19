from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.domain.model import ModelCode, ServingModelConfig
from app.inference.model_config_loader import ModelConfigLoader
from app.inference.model_loader import LoadedModel
from app.inference.model_registry import (
    ModelRegistry,
    ModelRegistryError,
)


@dataclass
class FakeModel:
    names: dict[int, str]
    predict_calls: list[dict[str, object]] = field(
        default_factory=list
    )

    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(dict(kwargs))
        return [object()]


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_model_paths(
    config: ServingModelConfig,
) -> dict[ModelCode, Path]:
    return {
        model.model_code: Path(model.file_path)
        for model in config.get_enabled_models()
    }


def create_loaded_models(
    config: ServingModelConfig,
    model_paths: dict[ModelCode, Path],
) -> dict[ModelCode, LoadedModel]:
    return {
        spec.model_code: LoadedModel(
            spec=spec,
            file_path=model_paths[spec.model_code],
            model=FakeModel(
                names={
                    class_id: class_spec.raw_name
                    for class_id, class_spec
                    in spec.classes.items()
                }
            ),
            device=config.execution.device,
            half_precision=(
                config.execution.half_precision
            ),
        )
        for spec in config.get_enabled_models()
    }


def test_registry_registers_enabled_models() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)
    registry = ModelRegistry()

    registry.register(
        config=config,
        model_paths=model_paths,
    )

    assert registry.is_configured is True
    assert registry.is_loaded is False
    assert registry.model_count == 3
    assert registry.loaded_model_count == 0

    registered_models = registry.list_models()

    assert tuple(
        registered.spec.model_code
        for registered in registered_models
    ) == (
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    )

    debris = registry.get(
        ModelCode.DEBRIS_DETECTOR
    )

    assert (
        debris.spec.model_code
        is ModelCode.DEBRIS_DETECTOR
    )
    assert debris.file_path == model_paths[
        ModelCode.DEBRIS_DETECTOR
    ]
    assert debris.model is None
    assert debris.device is None


def test_registry_attaches_loaded_models() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)
    loaded_models = create_loaded_models(
        config,
        model_paths,
    )

    registry = ModelRegistry()
    registry.register(
        config=config,
        model_paths=model_paths,
    )

    registry.attach_loaded_models(
        loaded_models
    )

    assert registry.is_configured is True
    assert registry.is_loaded is True
    assert registry.model_count == 3
    assert registry.loaded_model_count == 3

    debris = registry.get(
        ModelCode.DEBRIS_DETECTOR
    )

    assert (
        debris.model
        is loaded_models[
            ModelCode.DEBRIS_DETECTOR
        ].model
    )
    assert debris.device == "cuda:0"
    assert debris.half_precision is True


def test_registry_rejects_incomplete_loaded_models() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)
    loaded_models = create_loaded_models(
        config,
        model_paths,
    )

    loaded_models.pop(
        ModelCode.WILDLIFE_DETECTOR
    )

    registry = ModelRegistry()
    registry.register(
        config=config,
        model_paths=model_paths,
    )

    with pytest.raises(
        ModelRegistryError,
        match="WILDLIFE_DETECTOR",
    ):
        registry.attach_loaded_models(
            loaded_models
        )

    assert registry.is_loaded is False
    assert registry.loaded_model_count == 0


def test_registry_rejects_loaded_models_before_config() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    registry = ModelRegistry()

    with pytest.raises(
        ModelRegistryError,
        match="not configured",
    ):
        registry.attach_loaded_models(
            create_loaded_models(
                config,
                model_paths,
            )
        )


def test_registry_rejects_missing_model_path() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    model_paths.pop(
        ModelCode.WILDLIFE_DETECTOR
    )

    registry = ModelRegistry()

    with pytest.raises(
        ModelRegistryError,
        match="WILDLIFE_DETECTOR",
    ):
        registry.register(
            config=config,
            model_paths=model_paths,
        )

    assert registry.is_configured is False
    assert registry.model_count == 0


def test_registry_rejects_unexpected_disabled_model_path() -> None:
    config = load_model_config()

    models = [
        (
            model.model_copy(
                update={"enabled": False}
            )
            if (
                model.model_code
                is ModelCode.WILDLIFE_DETECTOR
            )
            else model
        )
        for model in config.models
    ]

    config_with_disabled_model = config.model_copy(
        update={"models": models}
    )

    model_paths = create_model_paths(config)
    registry = ModelRegistry()

    with pytest.raises(
        ModelRegistryError,
        match="WILDLIFE_DETECTOR",
    ):
        registry.register(
            config=config_with_disabled_model,
            model_paths=model_paths,
        )

    assert registry.is_configured is False
    assert registry.model_count == 0


def test_registry_get_rejects_unregistered_model() -> None:
    registry = ModelRegistry()

    with pytest.raises(
        ModelRegistryError,
        match="DEBRIS_DETECTOR",
    ):
        registry.get(
            ModelCode.DEBRIS_DETECTOR
        )


def test_registry_clear_removes_registered_models() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)
    registry = ModelRegistry()

    registry.register(
        config=config,
        model_paths=model_paths,
    )
    registry.attach_loaded_models(
        create_loaded_models(
            config,
            model_paths,
        )
    )

    registry.clear()

    assert registry.is_configured is False
    assert registry.is_loaded is False
    assert registry.model_count == 0
    assert registry.loaded_model_count == 0
    assert registry.list_models() == ()
