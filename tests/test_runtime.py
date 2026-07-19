from pathlib import Path

from app.core.runtime import (
    ApplicationRuntimeState,
    RuntimeStatus,
)
from app.domain.model import (
    ModelCode,
    ServingModelConfig,
)
from app.inference.model_config_loader import ModelConfigLoader


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_model_paths(
    config: ServingModelConfig,
) -> dict[ModelCode, Path]:
    return {
        model.model_code: Path(model.file_path)
        for model in config.get_enabled_models()
    }


def test_runtime_state_initial_values() -> None:
    runtime_state = ApplicationRuntimeState()

    assert runtime_state.status is RuntimeStatus.STARTING
    assert runtime_state.is_ready is False
    assert runtime_state.is_config_validated is False
    assert runtime_state.model_config is None
    assert runtime_state.model_paths == {}
    assert runtime_state.startup_error is None
    assert runtime_state.configured_model_count == 0
    assert runtime_state.validated_model_count == 0


def test_runtime_state_marks_configuration_validated() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    runtime_state = ApplicationRuntimeState()

    runtime_state.set_model_config(config)
    runtime_state.set_model_paths(model_paths)
    runtime_state.mark_config_validated()

    assert runtime_state.status is RuntimeStatus.CONFIG_VALIDATED
    assert runtime_state.is_config_validated is True
    assert runtime_state.is_ready is False
    assert runtime_state.configured_model_count == 3
    assert runtime_state.validated_model_count == 3
    assert runtime_state.startup_error is None

    assert set(runtime_state.model_paths) == {
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    }


def test_runtime_state_transitions_from_config_validated_to_ready() -> None:
    config = load_model_config()
    model_paths = create_model_paths(config)

    runtime_state = ApplicationRuntimeState()
    runtime_state.set_model_config(config)
    runtime_state.set_model_paths(model_paths)
    runtime_state.mark_config_validated()

    runtime_state.mark_ready()

    assert runtime_state.status is RuntimeStatus.READY
    assert runtime_state.is_config_validated is True
    assert runtime_state.is_ready is True
    assert runtime_state.configured_model_count == 3
    assert runtime_state.validated_model_count == 3
    assert runtime_state.startup_error is None


def test_runtime_state_records_startup_failure() -> None:
    runtime_state = ApplicationRuntimeState()

    runtime_state.mark_not_ready(
        RuntimeError("startup validation failed")
    )

    assert runtime_state.status is RuntimeStatus.NOT_READY
    assert runtime_state.is_config_validated is False
    assert runtime_state.is_ready is False
    assert runtime_state.startup_error == (
        "startup validation failed"
    )


def test_begin_startup_resets_previous_state() -> None:
    config = load_model_config()

    runtime_state = ApplicationRuntimeState()
    runtime_state.set_model_config(config)
    runtime_state.set_model_paths(
        {
            ModelCode.DEBRIS_DETECTOR: Path("debris.pt"),
        }
    )
    runtime_state.mark_not_ready("previous failure")

    runtime_state.begin_startup()

    assert runtime_state.status is RuntimeStatus.STARTING
    assert runtime_state.is_config_validated is False
    assert runtime_state.is_ready is False
    assert runtime_state.model_config is None
    assert runtime_state.model_paths == {}
    assert runtime_state.startup_error is None
    assert runtime_state.configured_model_count == 0
    assert runtime_state.validated_model_count == 0