from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.internal.errors import InternalAPIError
from app.main import create_app


@asynccontextmanager
async def empty_lifespan(
    _app: FastAPI,
) -> AsyncIterator[None]:
    yield


def test_internal_api_error_uses_standard_contract() -> None:
    app = create_app(
        lifespan_handler=empty_lifespan,
    )

    @app.get("/test/internal-error")
    async def raise_internal_error() -> None:
        raise InternalAPIError(
            status_code=401,
            code="AI_INTERNAL_AUTH_FAILED",
            message="내부 서버 인증에 실패했습니다.",
            trace_id=(
                "9ac3ead5-d26d-4c86-"
                "91ea-2c694c9f52fa"
            ),
        )

    with TestClient(app) as client:
        response = client.get("/test/internal-error")

    assert response.status_code == 401
    assert response.json() == {
        "success": False,
        "error": {
            "code": "AI_INTERNAL_AUTH_FAILED",
            "message": "내부 서버 인증에 실패했습니다.",
            "details": None,
        },
        "trace_id": (
            "9ac3ead5-d26d-4c86-"
            "91ea-2c694c9f52fa"
        ),
    }
