from dataclasses import dataclass
from hmac import compare_digest
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import Header

from app.api.internal.errors import InternalAPIError
from app.api.internal.schemas import InternalInferenceRequest
from app.core.config import settings


@dataclass(frozen=True)
class InternalRequestContext:
    idempotency_key: str
    request_id: str
    trace_id: str


def _create_response_trace_id(
    trace_id: str | None,
) -> str:
    if trace_id:
        try:
            return str(UUID(trace_id))
        except ValueError:
            pass

    return str(uuid4())


def _require_uuid_header(
    value: str | None,
    *,
    header_name: str,
    trace_id: str,
) -> str:
    if value is None:
        raise InternalAPIError(
            status_code=400,
            code="COMMON_BAD_REQUEST",
            message="잘못된 요청입니다.",
            trace_id=trace_id,
            details={
                "missing_header": header_name,
            },
        )

    try:
        return str(UUID(value))
    except ValueError as exc:
        raise InternalAPIError(
            status_code=400,
            code="COMMON_BAD_REQUEST",
            message="잘못된 요청입니다.",
            trace_id=trace_id,
            details={
                "invalid_header": header_name,
            },
        ) from exc


async def require_internal_request(
    x_internal_api_key: Annotated[
        str | None,
        Header(alias="X-Internal-API-Key"),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Header(alias="Idempotency-Key"),
    ] = None,
    x_request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID"),
    ] = None,
    x_trace_id: Annotated[
        str | None,
        Header(alias="X-Trace-ID"),
    ] = None,
) -> InternalRequestContext:
    response_trace_id = _create_response_trace_id(
        x_trace_id
    )
    configured_key = settings.internal_api_key

    if (
        configured_key is None
        or x_internal_api_key is None
        or not compare_digest(
            x_internal_api_key,
            configured_key.get_secret_value(),
        )
    ):
        raise InternalAPIError(
            status_code=401,
            code="AI_INTERNAL_AUTH_FAILED",
            message="AI 내부 서버 인증에 실패했습니다.",
            trace_id=response_trace_id,
        )

    validated_trace_id = _require_uuid_header(
        x_trace_id,
        header_name="X-Trace-ID",
        trace_id=response_trace_id,
    )
    validated_request_id = _require_uuid_header(
        x_request_id,
        header_name="X-Request-ID",
        trace_id=validated_trace_id,
    )
    validated_idempotency_key = _require_uuid_header(
        idempotency_key,
        header_name="Idempotency-Key",
        trace_id=validated_trace_id,
    )

    return InternalRequestContext(
        idempotency_key=validated_idempotency_key,
        request_id=validated_request_id,
        trace_id=validated_trace_id,
    )



def validate_request_id_matches_body(
    context: InternalRequestContext,
    request: InternalInferenceRequest,
) -> None:
    if context.request_id == str(request.request_id):
        return

    raise InternalAPIError(
        status_code=400,
        code="COMMON_BAD_REQUEST",
        message="잘못된 요청입니다.",
        trace_id=context.trace_id,
        details={
            "mismatched_fields": [
                "X-Request-ID",
                "request_id",
            ],
        },
    )
