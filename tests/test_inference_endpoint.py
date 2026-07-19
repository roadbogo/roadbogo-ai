from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.endpoints import inference
from app.domain.model import ModelCode, ServingModelConfig
from app.inference.model_config_loader import ModelConfigLoader
from app.inference.model_loader import LoadedModel
from app.inference.model_registry import ModelRegistry
from app.main import create_app


@dataclass
class FakeBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class FakeResult:
    boxes: FakeBoxes | None


@dataclass
class FakeModel:
    names: dict[int, str]
    result: FakeResult

    def predict(
        self,
        **kwargs: object,
    ) -> list[FakeResult]:
        return [self.result]


def load_model_config() -> ServingModelConfig:
    return ModelConfigLoader().load()


def create_loaded_registry() -> ModelRegistry:
    config = load_model_config()

    model_paths = {
        spec.model_code: Path(spec.file_path)
        for spec in config.get_enabled_models()
    }

    loaded_models: dict[
        ModelCode,
        LoadedModel,
    ] = {}

    for spec in config.get_enabled_models():
        if (
            spec.model_code
            is ModelCode.DEBRIS_DETECTOR
        ):
            result = FakeResult(
                boxes=FakeBoxes(
                    xyxy=np.array(
                        [[1.0, 2.0, 12.0, 8.0]]
                    ),
                    conf=np.array([0.93]),
                    cls=np.array([3.0]),
                )
            )
        else:
            result = FakeResult(boxes=None)

        fake_model = FakeModel(
            names={
                class_id: class_spec.raw_name
                for class_id, class_spec
                in spec.classes.items()
            },
            result=result,
        )

        loaded_models[spec.model_code] = LoadedModel(
            spec=spec,
            file_path=model_paths[spec.model_code],
            model=fake_model,
            device=config.execution.device,
            half_precision=(
                config.execution.half_precision
            ),
        )

    registry = ModelRegistry()
    registry.register(
        config=config,
        model_paths=model_paths,
    )
    registry.attach_loaded_models(
        loaded_models
    )

    return registry


def create_test_image() -> bytes:
    image = np.zeros(
        (10, 20, 3),
        dtype=np.uint8,
    )

    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    assert success is True

    return encoded.tobytes()


def create_test_app(
    registry: ModelRegistry,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        app.state.model_registry = registry
        yield

    return create_app(
        lifespan_handler=lifespan,
    )


def test_inference_endpoint_returns_detections() -> None:
    app = create_test_app(
        create_loaded_registry()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/",
            files={
                "file": (
                    "sample.png",
                    create_test_image(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["image_width"] == 20
    assert body["image_height"] == 10
    assert body["model_count"] == 3
    assert body["total_detection_count"] == 1
    assert body["incident_detection_count"] == 1

    debris_result = body["model_results"][0]

    assert debris_result["model_code"] == (
        "DEBRIS_DETECTOR"
    )
    assert debris_result["detection_count"] == 1

    detection = debris_result["detections"][0]

    assert detection["class_code"] == "TIRE"
    assert detection["confidence"] == 0.93
    assert detection["incident_target"] is True


def test_inference_endpoint_rejects_invalid_image() -> None:
    app = create_test_app(
        create_loaded_registry()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/",
            files={
                "file": (
                    "broken.png",
                    b"not-an-image",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Uploaded file is not a valid image."
        )
    }


def test_inference_endpoint_rejects_media_type() -> None:
    app = create_test_app(
        create_loaded_registry()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/",
            files={
                "file": (
                    "sample.gif",
                    b"gif-data",
                    "image/gif",
                )
            },
        )

    assert response.status_code == 415
    assert response.json() == {
        "detail": (
            "Only JPEG and PNG image files are supported."
        )
    }


def test_inference_endpoint_rejects_unready_models() -> None:
    app = create_test_app(
        ModelRegistry()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/",
            files={
                "file": (
                    "sample.png",
                    create_test_image(),
                    "image/png",
                )
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Inference models are not ready."
    }


def test_inference_endpoint_rejects_large_image(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        inference,
        "MAX_IMAGE_BYTES",
        4,
    )

    app = create_test_app(
        create_loaded_registry()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/",
            files={
                "file": (
                    "sample.png",
                    b"12345",
                    "image/png",
                )
            },
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": (
            "Uploaded image exceeds the maximum size of "
            "4 bytes."
        )
    }
