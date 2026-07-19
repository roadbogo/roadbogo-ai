from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response, status

from app.core.config import settings
from app.core.runtime import (
    ApplicationRuntimeState,
    RuntimeStatus,
)
from app.inference.model_registry import ModelRegistry


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


def create_utc_timestamp() -> str:
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
)
async def liveness_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "timestamp": create_utc_timestamp(),
    }


@router.get("/ready")
async def readiness_check(
    request: Request,
    response: Response,
) -> dict[str, object]:
    runtime_state: ApplicationRuntimeState | None = getattr(
        request.app.state,
        "runtime_state",
        None,
    )
    model_registry: ModelRegistry | None = getattr(
        request.app.state,
        "model_registry",
        None,
    )

    if runtime_state is None or model_registry is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return {
            "status": "not_ready",
            "service": settings.app_name,
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

    is_ready = (
        runtime_state.is_ready
        and model_registry.is_loaded
    )

    response.status_code = (
        status.HTTP_200_OK
        if is_ready
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    result: dict[str, object] = {
        "status": "ready" if is_ready else "not_ready",
        "service": settings.app_name,
        "runtime_status": runtime_state.status.value,
        "config_validated": (
            runtime_state.is_config_validated
        ),
        "inference_ready": is_ready,
        "configured_model_count": (
            runtime_state.configured_model_count
        ),
        "validated_model_count": (
            runtime_state.validated_model_count
        ),
        "registered_model_count": (
            model_registry.model_count
        ),
        "loaded_model_count": (
            model_registry.loaded_model_count
        ),
    }

    if runtime_state.status is RuntimeStatus.CONFIG_VALIDATED:
        result["detail"] = (
            "Model configuration and files are validated, "
            "but YOLO model and GPU initialization are not complete."
        )
    elif runtime_state.startup_error is not None:
        if runtime_state.is_config_validated:
            result["detail"] = (
                "Model configuration was validated, "
                "but inference initialization failed."
            )
        else:
            result["detail"] = "Startup validation failed."
    elif not is_ready:
        result["detail"] = (
            "Inference initialization is not complete."
        )

    return result
