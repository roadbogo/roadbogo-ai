from pathlib import Path

import pytest

from app.domain.model import ModelCode, ServingModelConfig
from app.inference.model_config_loader import ModelConfigLoader
from app.inference.model_registry import (
    ModelRegistry,
    ModelRegistryError,
)


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_model_paths(
    config: ServingModelConfig,
) -> dict[ModelCode, Path]:
    return {
        model.model_code: Path(model.file_path)
        for model in config.get_enabled_models()
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
    assert registry.model_count == 3

    registered_models = registry.list_models()

    assert tuple(
        registered.spec.model_code
        for registered in registered_models
    ) == (
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    )

    debris = registry.get(ModelCode.DEBRIS_DETECTOR)

    assert debris.spec.model_code is ModelCode.DEBRIS_DETECTOR
    assert debris.file_path == model_paths[
        ModelCode.DEBRIS_DETECTOR
    ]


def test_registry_rejects_missing_model_path() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    model_paths.pop(ModelCode.WILDLIFE_DETECTOR)

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
            model.model_copy(update={"enabled": False})
            if model.model_code is ModelCode.WILDLIFE_DETECTOR
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
        registry.get(ModelCode.DEBRIS_DETECTOR)


def test_registry_clear_removes_registered_models() -> None:
    config = load_model_config()
    registry = ModelRegistry()

    registry.register(
        config=config,
        model_paths=create_model_paths(config),
    )

    registry.clear()

    assert registry.is_configured is False
    assert registry.model_count == 0
    assert registry.list_models() == ()