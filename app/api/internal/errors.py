from typing import Any
from uuid import UUID, uuid4

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings


class InternalAPIError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        trace_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.trace_id = trace_id
        self.details = details


def _get_request_trace_id(
    request: Request,
) -> str:
    header_value = request.headers.get("X-Trace-ID")

    if header_value:
        try:
            return str(UUID(header_value))
        except ValueError:
            pass

    return str(uuid4())


async def internal_api_error_handler(
    _request: Request,
    exc: InternalAPIError,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
            "trace_id": exc.trace_id,
        },
    )


async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    validation_errors = jsonable_encoder(
        exc.errors()
    )

    if request.url.path.startswith(
        settings.internal_api_v1_prefix
    ):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "COMMON_BAD_REQUEST",
                    "message": "잘못된 요청입니다.",
                    "details": {
                        "validation_errors": (
                            validation_errors
                        ),
                    },
                },
                "trace_id": _get_request_trace_id(
                    request
                ),
            },
        )

    return JSONResponse(
        status_code=422,
        content={
            "detail": validation_errors,
        },
    )
