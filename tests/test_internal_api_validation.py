from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.internal.schemas import (
    InternalInferenceRequest,
)
from app.core.config import settings
from app.main import create_app


TRACE_ID = (
    "9ac3ead5-d26d-4c86-91ea-2c694c9f52fa"
)


@asynccontextmanager
async def empty_lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    yield


class PublicTestRequest(BaseModel):
    value: int


def test_internal_validation_uses_standard_error() -> None:
    app = create_app(
        lifespan_handler=empty_lifespan,
    )

    @app.post(
        f"{settings.internal_api_v1_prefix}"
        "/test-validation"
    )
    async def validate_internal_request(
        _request: InternalInferenceRequest,
    ) -> None:
        return None

    with TestClient(app) as client:
        response = client.post(
            (
                f"{settings.internal_api_v1_prefix}"
                "/test-validation"
            ),
            headers={
                "X-Trace-ID": TRACE_ID,
            },
            json={
                "request_id": "not-a-uuid",
            },
        )

    body = response.json()

    assert response.status_code == 400
    assert body["success"] is False
    assert body["error"]["code"] == (
        "COMMON_BAD_REQUEST"
    )
    assert body["error"]["message"] == (
        "잘못된 요청입니다."
    )
    assert body["trace_id"] == TRACE_ID

    validation_errors = body["error"]["details"][
        "validation_errors"
    ]

    assert validation_errors
    assert validation_errors[0]["loc"][0] == "body"


def test_public_validation_keeps_fastapi_contract() -> None:
    app = create_app(
        lifespan_handler=empty_lifespan,
    )

    @app.post("/test-public-validation")
    async def validate_public_request(
        _request: PublicTestRequest,
    ) -> None:
        return None

    with TestClient(app) as client:
        response = client.post(
            "/test-public-validation",
            json={
                "value": "not-an-integer",
            },
        )

    assert response.status_code == 422
    assert "detail" in response.json()
    assert "success" not in response.json()
