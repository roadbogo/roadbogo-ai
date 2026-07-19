from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.lifespan import create_lifespan
from app.core.runtime import ApplicationRuntimeState
from app.domain.model import ModelCode, ServingModelConfig
from app.inference.model_config_loader import (
    ModelConfigError,
    ModelConfigLoader,
)
from app.inference.model_loader import (
    LoadedModel,
    ModelLoaderError,
)
from app.inference.model_registry import ModelRegistry
from app.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


@dataclass
class FakeModel:
    names: dict[int, str]

    def predict(self, **kwargs: object) -> list[object]:
        return [object()]


class StubModelConfigLoader:
    def __init__(
        self,
        *,
        config: ServingModelConfig | None = None,
        model_paths: dict[ModelCode, Path] | None = None,
        error: ModelConfigError | None = None,
    ) -> None:
        self.config = config
        self.model_paths = model_paths or {}
        self.error = error

    def load(self) -> ServingModelConfig:
        if self.error is not None:
            raise self.error

        if self.config is None:
            raise AssertionError(
                "Test model configuration was not provided."
            )

        return self.config

    def validate_model_files(
        self,
        config: ServingModelConfig,
    ) -> dict[ModelCode, Path]:
        if config is not self.config:
            raise AssertionError(
                "Unexpected model configuration instance."
            )

        return dict(self.model_paths)


class StubModelLoader:
    def __init__(
        self,
        *,
        loaded_models: dict[
            ModelCode,
            LoadedModel,
        ] | None = None,
        error: ModelLoaderError | None = None,
    ) -> None:
        self.loaded_models = loaded_models or {}
        self.error = error

    def load(
        self,
        config: ServingModelConfig,
        model_paths: dict[ModelCode, Path],
    ) -> dict[ModelCode, LoadedModel]:
        if self.error is not None:
            raise self.error

        expected_codes = {
            model.model_code
            for model in config.get_enabled_models()
        }

        if set(model_paths) != expected_codes:
            raise AssertionError(
                "Unexpected model paths."
            )

        return dict(self.loaded_models)


def load_verified_config() -> ServingModelConfig:
    return ModelConfigLoader(
        config_path=CONFIG_PATH,
        project_root=PROJECT_ROOT,
    ).load()


def create_temporary_model_paths(
    tmp_path: Path,
    config: ServingModelConfig,
) -> dict[ModelCode, Path]:
    model_paths: dict[ModelCode, Path] = {}

    for model in config.get_enabled_models():
        model_path = (
            tmp_path
            / f"{model.model_code.value.lower()}.pt"
        )
        model_path.write_bytes(b"test model file")
        model_paths[model.model_code] = model_path

    return model_paths


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


def test_health_reports_inference_ready(
    tmp_path: Path,
) -> None:
    config = load_verified_config()
    model_paths = create_temporary_model_paths(
        tmp_path,
        config,
    )

    app = create_app(
        lifespan_handler=create_lifespan(
            StubModelConfigLoader(
                config=config,
                model_paths=model_paths,
            ),
            StubModelLoader(
                loaded_models=create_loaded_models(
                    config,
                    model_paths,
                )
            ),
        ),
    )

    with TestClient(app) as client:
        live_response = client.get("/health/live")
        ready_response = client.get("/health/ready")
        legacy_response = client.get("/health")

    assert live_response.status_code == 200

    live_body = live_response.json()

    assert live_body["status"] == "ok"
    assert live_body["service"] == "roadbogo-ai"
    assert live_body["timestamp"].endswith("Z")

    parsed_timestamp = datetime.fromisoformat(
        live_body["timestamp"].replace(
            "Z",
            "+00:00",
        )
    )

    assert parsed_timestamp.tzinfo is not None
    assert (
        parsed_timestamp.utcoffset().total_seconds()
        == 0
    )

    assert ready_response.status_code == 200
    assert ready_response.json() == {
        "status": "ready",
        "service": "roadbogo-ai",
        "runtime_status": "ready",
        "config_validated": True,
        "inference_ready": True,
        "configured_model_count": 3,
        "validated_model_count": 3,
        "registered_model_count": 3,
        "loaded_model_count": 3,
    }

    assert legacy_response.status_code == 200
    assert legacy_response.json()["status"] == "ok"


def test_health_reports_config_validated_state(
    tmp_path: Path,
) -> None:
    config = load_verified_config()
    model_paths = create_temporary_model_paths(
        tmp_path,
        config,
    )

    @asynccontextmanager
    async def config_validated_lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        runtime_state = ApplicationRuntimeState()
        model_registry = ModelRegistry()

        model_registry.register(
            config=config,
            model_paths=model_paths,
        )
        runtime_state.set_model_config(config)
        runtime_state.set_model_paths(model_paths)
        runtime_state.mark_config_validated()

        app.state.runtime_state = runtime_state
        app.state.model_registry = model_registry

        yield

        model_registry.clear()

    app = create_app(
        lifespan_handler=config_validated_lifespan,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "roadbogo-ai",
        "runtime_status": "config_validated",
        "config_validated": True,
        "inference_ready": False,
        "configured_model_count": 3,
        "validated_model_count": 3,
        "registered_model_count": 3,
        "loaded_model_count": 0,
        "detail": (
            "Model configuration and files are validated, "
            "but YOLO model and GPU initialization are not complete."
        ),
    }


def test_readiness_reports_model_initialization_failure(
    tmp_path: Path,
) -> None:
    config = load_verified_config()
    model_paths = create_temporary_model_paths(
        tmp_path,
        config,
    )

    app = create_app(
        lifespan_handler=create_lifespan(
            StubModelConfigLoader(
                config=config,
                model_paths=model_paths,
            ),
            StubModelLoader(
                error=ModelLoaderError(
                    "test model initialization failure"
                )
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")
        runtime_error = (
            client.app.state.runtime_state.startup_error
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "roadbogo-ai",
        "runtime_status": "not_ready",
        "config_validated": True,
        "inference_ready": False,
        "configured_model_count": 3,
        "validated_model_count": 3,
        "registered_model_count": 3,
        "loaded_model_count": 0,
        "detail": (
            "Model configuration was validated, "
            "but inference initialization failed."
        ),
    }
    assert runtime_error == (
        "test model initialization failure"
    )


def test_readiness_reports_config_validation_failure() -> None:
    app = create_app(
        lifespan_handler=create_lifespan(
            StubModelConfigLoader(
                error=ModelConfigError(
                    "test startup failure"
                ),
            ),
            StubModelLoader(),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")
        runtime_error = (
            client.app.state.runtime_state.startup_error
        )

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "roadbogo-ai",
        "runtime_status": "not_ready",
        "config_validated": False,
        "inference_ready": False,
        "configured_model_count": 0,
        "validated_model_count": 0,
        "registered_model_count": 0,
        "loaded_model_count": 0,
        "detail": "Startup validation failed.",
    }
    assert runtime_error == "test startup failure"


def test_readiness_reports_uninitialized_runtime() -> None:
    @asynccontextmanager
    async def empty_lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        yield

    app = create_app(
        lifespan_handler=empty_lifespan,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "roadbogo-ai",
        "runtime_status": "not_initialized",
        "config_validated": False,
        "inference_ready": False,
        "configured_model_count": 0,
        "validated_model_count": 0,
        "registered_model_count": 0,
        "loaded_model_count": 0,
        "detail": (
            "Application runtime state is not initialized."
        ),
    }


def test_openapi_contains_health_paths() -> None:
    app = create_app()

    health_paths = {
        path
        for path in app.openapi()["paths"]
        if path.startswith("/health")
    }

    assert health_paths == {
        "/health",
        "/health/live",
        "/health/ready",
    }
