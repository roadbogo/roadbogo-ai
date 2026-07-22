from uuid import uuid4

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.internal import inferences
from app.api.internal.errors import (
    InternalAPIError,
    internal_api_error_handler,
    request_validation_error_handler,
)
from app.api.internal.router import internal_api_router
from app.core.config import settings
from app.domain.model import ModelCode
from app.main import create_app
from app.schemas.detection import (
    BoundingBox,
    Detection,
    ImageInferenceResponse,
    ModelDetectionResult,
)
from app.services.image_download_service import (
    ImageDownloadInputError,
)


def create_internal_test_app() -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(
        InternalAPIError,
        internal_api_error_handler,
    )
    app.add_exception_handler(
        RequestValidationError,
        request_validation_error_handler,
    )
    app.include_router(
        internal_api_router,
        prefix="/api/internal/v1",
    )
    app.state.model_registry = object()

    return app


def create_payload(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "inference_run_public_id": str(uuid4()),
        "video_frame": {
            "public_id": str(uuid4()),
            "cctv_public_id": str(uuid4()),
            "captured_at": "2026-07-22T01:00:00Z",
            "frame_sequence": 10,
            "original_width": 1920,
            "original_height": 1080,
            "capture_source": "LIVE",
        },
        "input": {
            "file_public_id": str(uuid4()),
            "download_url": (
                "https://storage.example/frame.png"
            ),
            "mime_type": "image/png",
        },
        "tracking": {
            "tracking_session_key": "cctv-session-1",
        },
        "execution": {
            "model_codes": [
                "DEBRIS_DETECTOR",
            ],
            "execution_mode": "SEQUENTIAL",
            "threshold_profile_code": "DEFAULT",
        },
    }


def create_headers(
    request_id: str,
    trace_id: str,
) -> dict[str, str]:
    return {
        "X-Internal-API-Key": "internal-secret",
        "Idempotency-Key": str(uuid4()),
        "X-Request-ID": request_id,
        "X-Trace-ID": trace_id,
    }


def test_internal_inference_route_is_registered() -> None:
    paths = create_app().openapi()["paths"]

    assert "/api/internal/v1/inferences" in paths


def test_internal_inference_runs_selected_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr("internal-secret"),
    )
    monkeypatch.setattr(
        inferences.image_download_service,
        "download",
        lambda *_args, **_kwargs: b"image-bytes",
    )

    captured = {}

    def predict(
        image_bytes,
        model_registry,
        *,
        model_codes,
    ):
        captured["image_bytes"] = image_bytes
        captured["model_registry"] = model_registry
        captured["model_codes"] = model_codes

        return ImageInferenceResponse(
            image_width=100,
            image_height=50,
            model_count=1,
            total_detection_count=1,
            incident_detection_count=1,
            model_results=[
                ModelDetectionResult(
                    model_code=ModelCode.DEBRIS_DETECTOR,
                    display_name="Debris",
                    detection_count=1,
                    detections=[
                        Detection(
                            class_id=3,
                            raw_name="tire",
                            class_code="TIRE",
                            incident_target=True,
                            confidence=0.91,
                            bounding_box=BoundingBox(
                                x_min=10,
                                y_min=10,
                                x_max=50,
                                y_max=30,
                            ),
                        )
                    ],
                )
            ],
        )

    monkeypatch.setattr(
        inferences.detection_service,
        "predict",
        predict,
    )

    request_id = str(uuid4())
    trace_id = str(uuid4())

    response = TestClient(
        create_internal_test_app()
    ).post(
        "/api/internal/v1/inferences",
        json=create_payload(request_id),
        headers=create_headers(
            request_id,
            trace_id,
        ),
    )

    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    assert body["trace_id"] == trace_id
    assert body["data"]["request_id"] == request_id
    assert body["data"]["image"] == {
        "width": 100,
        "height": 50,
    }
    assert body["data"]["model_count"] == 1
    assert body["data"]["total_detection_count"] == 1

    model_result = body["data"]["model_results"][0]

    assert model_result["model_code"] == "DEBRIS_DETECTOR"

    detection = model_result["detections"][0]

    assert detection["detection_index"] == 0
    assert detection["class_index"] == 3
    assert detection["bounding_box"] == {
        "x": 0.1,
        "y": 0.2,
        "width": 0.4,
        "height": 0.4,
    }

    assert captured["image_bytes"] == b"image-bytes"
    assert captured["model_codes"] == [
        ModelCode.DEBRIS_DETECTOR,
    ]


def test_internal_inference_maps_download_input_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr("internal-secret"),
    )

    def reject_download(*_args, **_kwargs):
        raise ImageDownloadInputError(
            "Downloaded image MIME type does not match."
        )

    monkeypatch.setattr(
        inferences.image_download_service,
        "download",
        reject_download,
    )

    request_id = str(uuid4())
    trace_id = str(uuid4())

    response = TestClient(
        create_internal_test_app()
    ).post(
        "/api/internal/v1/inferences",
        json=create_payload(request_id),
        headers=create_headers(
            request_id,
            trace_id,
        ),
    )

    assert response.status_code == 400
    body = response.json()

    assert body["success"] is False
    assert body["error"]["code"] == "AI_INPUT_INVALID"
    assert body["trace_id"] == trace_id