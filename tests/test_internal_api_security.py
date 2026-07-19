from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.internal.security import (
    InternalRequestContext,
    require_internal_request,
)
from app.core.config import settings
from app.main import create_app


INTERNAL_API_KEY = "test-internal-secret"
IDEMPOTENCY_KEY = (
    "025c928c-86d3-47af-950a-d66aa1bed881"
)
REQUEST_ID = (
    "d19cc973-167b-40a1-95f0-82760d70aa23"
)
TRACE_ID = (
    "9ac3ead5-d26d-4c86-91ea-2c694c9f52fa"
)


@asynccontextmanager
async def empty_lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    yield


def create_internal_test_app() -> FastAPI:
    app = create_app(
        lifespan_handler=empty_lifespan,
    )

    @app.post("/test/internal-request")
    async def internal_request(
        context: Annotated[
            InternalRequestContext,
            Depends(require_internal_request),
        ],
    ) -> dict[str, str]:
        return {
            "idempotency_key": (
                context.idempotency_key
            ),
            "request_id": context.request_id,
            "trace_id": context.trace_id,
        }

    return app


def create_valid_headers() -> dict[str, str]:
    return {
        "X-Internal-API-Key": INTERNAL_API_KEY,
        "Idempotency-Key": IDEMPOTENCY_KEY,
        "X-Request-ID": REQUEST_ID,
        "X-Trace-ID": TRACE_ID,
    }


def test_internal_request_accepts_valid_headers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr(INTERNAL_API_KEY),
    )
    app = create_internal_test_app()

    with TestClient(app) as client:
        response = client.post(
            "/test/internal-request",
            headers=create_valid_headers(),
        )

    assert response.status_code == 200
    assert response.json() == {
        "idempotency_key": IDEMPOTENCY_KEY,
        "request_id": REQUEST_ID,
        "trace_id": TRACE_ID,
    }


def test_internal_request_rejects_invalid_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr(INTERNAL_API_KEY),
    )
    app = create_internal_test_app()
    headers = create_valid_headers()
    headers["X-Internal-API-Key"] = "wrong-secret"

    with TestClient(app) as client:
        response = client.post(
            "/test/internal-request",
            headers=headers,
        )

    assert response.status_code == 401
    assert response.json() == {
        "success": False,
        "error": {
            "code": "AI_INTERNAL_AUTH_FAILED",
            "message": (
                "AI 내부 서버 인증에 실패했습니다."
            ),
            "details": None,
        },
        "trace_id": TRACE_ID,
    }


def test_internal_request_rejects_missing_header(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr(INTERNAL_API_KEY),
    )
    app = create_internal_test_app()
    headers = create_valid_headers()
    del headers["X-Request-ID"]

    with TestClient(app) as client:
        response = client.post(
            "/test/internal-request",
            headers=headers,
        )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": {
            "code": "COMMON_BAD_REQUEST",
            "message": "잘못된 요청입니다.",
            "details": {
                "missing_header": "X-Request-ID",
            },
        },
        "trace_id": TRACE_ID,
    }


def test_internal_request_rejects_invalid_uuid_header(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "internal_api_key",
        SecretStr(INTERNAL_API_KEY),
    )
    app = create_internal_test_app()
    headers = create_valid_headers()
    headers["Idempotency-Key"] = "not-a-uuid"

    with TestClient(app) as client:
        response = client.post(
            "/test/internal-request",
            headers=headers,
        )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": {
            "code": "COMMON_BAD_REQUEST",
            "message": "잘못된 요청입니다.",
            "details": {
                "invalid_header": (
                    "Idempotency-Key"
                ),
            },
        },
        "trace_id": TRACE_ID,
    }
