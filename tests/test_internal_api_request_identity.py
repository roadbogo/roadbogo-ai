import pytest

from app.api.internal.errors import InternalAPIError
from app.api.internal.schemas import (
    InternalInferenceRequest,
)
from app.api.internal.security import (
    InternalRequestContext,
    validate_request_id_matches_body,
)


REQUEST_ID = (
    "2ba33b26-016e-4c28-acaf-58366a652b42"
)
TRACE_ID = (
    "9ac3ead5-d26d-4c86-91ea-2c694c9f52fa"
)

VALID_REQUEST = {
    "request_id": REQUEST_ID,
    "inference_run_public_id": (
        "22988e7d-8fad-4c9e-9bd2-087d83a61e47"
    ),
    "video_frame": {
        "public_id": (
            "b1603567-b684-4425-8436-750ff67ef65d"
        ),
        "cctv_public_id": (
            "64da7102-b643-453f-a17c-f4dd45d07561"
        ),
        "captured_at": (
            "2026-07-13T10:20:32.125Z"
        ),
        "frame_sequence": 1520,
        "original_width": 1920,
        "original_height": 1080,
        "capture_source": "LIVE",
    },
    "input": {
        "file_public_id": (
            "23b72c92-c4aa-4bde-ad0f-157173f027a1"
        ),
        "download_url": (
            "https://api.internal.example/"
            "files/example/content?token=test"
        ),
        "mime_type": "image/jpeg",
    },
    "tracking": {
        "tracking_session_key": (
            "CCTV-00015-20260713-001"
        ),
    },
    "execution": {
        "model_codes": [
            "VEHICLE_DETECTOR",
            "DEBRIS_DETECTOR",
            "WILDLIFE_DETECTOR",
        ],
        "execution_mode": "PARALLEL",
        "threshold_profile_code": "MVP_DEFAULT",
    },
}


def create_context(
    request_id: str,
) -> InternalRequestContext:
    return InternalRequestContext(
        idempotency_key=(
            "025c928c-86d3-47af-"
            "950a-d66aa1bed881"
        ),
        request_id=request_id,
        trace_id=TRACE_ID,
    )


def test_request_id_matches_body() -> None:
    request = InternalInferenceRequest.model_validate(
        VALID_REQUEST
    )

    validate_request_id_matches_body(
        create_context(REQUEST_ID),
        request,
    )


def test_request_id_mismatch_is_rejected() -> None:
    request = InternalInferenceRequest.model_validate(
        VALID_REQUEST
    )

    with pytest.raises(InternalAPIError) as exc_info:
        validate_request_id_matches_body(
            create_context(
                "d19cc973-167b-40a1-"
                "95f0-82760d70aa23"
            ),
            request,
        )

    error = exc_info.value

    assert error.status_code == 400
    assert error.code == "COMMON_BAD_REQUEST"
    assert error.message == "잘못된 요청입니다."
    assert error.trace_id == TRACE_ID
    assert error.details == {
        "mismatched_fields": [
            "X-Request-ID",
            "request_id",
        ],
    }
