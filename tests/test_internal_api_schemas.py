from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.api.internal.schemas import (
    ContractExecutionMode,
    ContractModelCode,
    InternalInferenceRequest,
)
from app.domain.model import ModelCode


VALID_REQUEST = {
    "request_id": (
        "2ba33b26-016e-4c28-acaf-58366a652b42"
    ),
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


def test_internal_inference_request_accepts_contract() -> None:
    request = InternalInferenceRequest.model_validate(
        VALID_REQUEST
    )

    assert request.execution.execution_mode is (
        ContractExecutionMode.PARALLEL
    )
    assert request.execution.model_codes == [
        ContractModelCode.VEHICLE_DETECTOR,
        ContractModelCode.DEBRIS_DETECTOR,
        ContractModelCode.WILDLIFE_DETECTOR,
    ]


def test_vehicle_contract_code_maps_to_internal_model() -> None:
    request = InternalInferenceRequest.model_validate(
        VALID_REQUEST
    )

    assert request.execution.get_internal_model_codes() == [
        ModelCode.PROHIBITED_MOBILITY_DETECTOR,
        ModelCode.DEBRIS_DETECTOR,
        ModelCode.WILDLIFE_DETECTOR,
    ]


def test_internal_model_code_is_not_exposed_in_contract() -> None:
    payload = deepcopy(VALID_REQUEST)
    payload["execution"]["model_codes"] = [
        "PROHIBITED_MOBILITY_DETECTOR"
    ]

    with pytest.raises(ValidationError):
        InternalInferenceRequest.model_validate(payload)


def test_captured_at_must_be_utc() -> None:
    payload = deepcopy(VALID_REQUEST)
    payload["video_frame"]["captured_at"] = (
        "2026-07-13T19:20:32.125+09:00"
    )

    with pytest.raises(
        ValidationError,
        match="captured_at must use ISO 8601 UTC",
    ):
        InternalInferenceRequest.model_validate(payload)


def test_model_codes_must_not_be_empty() -> None:
    payload = deepcopy(VALID_REQUEST)
    payload["execution"]["model_codes"] = []

    with pytest.raises(ValidationError):
        InternalInferenceRequest.model_validate(payload)


def test_model_codes_must_not_have_duplicates() -> None:
    payload = deepcopy(VALID_REQUEST)
    payload["execution"]["model_codes"] = [
        "DEBRIS_DETECTOR",
        "DEBRIS_DETECTOR",
    ]

    with pytest.raises(
        ValidationError,
        match="model_codes must not contain duplicates",
    ):
        InternalInferenceRequest.model_validate(payload)


def test_input_mime_type_must_be_supported() -> None:
    payload = deepcopy(VALID_REQUEST)
    payload["input"]["mime_type"] = "image/gif"

    with pytest.raises(
        ValidationError,
        match="Only JPEG and PNG",
    ):
        InternalInferenceRequest.model_validate(payload)
