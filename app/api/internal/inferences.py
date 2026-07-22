from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from app.api.internal.errors import InternalAPIError
from app.api.internal.schemas import (
    CONTRACT_TO_INTERNAL_MODEL_CODE,
    InternalBoundingBoxResponse,
    InternalDetectionResponse,
    InternalImageResponse,
    InternalInferenceData,
    InternalInferenceRequest,
    InternalInferenceResponse,
    InternalModelResultResponse,
)
from app.api.internal.security import (
    InternalRequestContext,
    require_internal_request,
    validate_request_id_matches_body,
)
from app.inference.model_registry import ModelRegistry
from app.schemas.detection import (
    ImageInferenceResponse,
    ModelDetectionResult,
)
from app.services.detection_service import (
    DetectionInputError,
    DetectionService,
    DetectionServiceError,
    DetectionUnavailableError,
)
from app.services.image_download_service import (
    ImageDownloadInputError,
    ImageDownloadService,
    ImageDownloadUnavailableError,
)


router = APIRouter()
image_download_service = ImageDownloadService()
detection_service = DetectionService()


def _normalize_axis(
    start: float,
    end: float,
    total: int,
) -> tuple[float, float]:
    normalized_start = round(
        min(max(start / total, 0.0), 1.0),
        7,
    )
    normalized_end = round(
        min(max(end / total, 0.0), 1.0),
        7,
    )
    normalized_size = round(
        max(normalized_end - normalized_start, 0.0),
        7,
    )

    if normalized_start + normalized_size > 1.0:
        normalized_size = round(
            max(1.0 - normalized_start, 0.0),
            7,
        )

    return normalized_start, normalized_size


def _create_bounding_box_response(
    result: ImageInferenceResponse,
    detection,
) -> InternalBoundingBoxResponse:
    x, width = _normalize_axis(
        detection.bounding_box.x_min,
        detection.bounding_box.x_max,
        result.image_width,
    )
    y, height = _normalize_axis(
        detection.bounding_box.y_min,
        detection.bounding_box.y_max,
        result.image_height,
    )

    return InternalBoundingBoxResponse(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _create_model_result_response(
    *,
    contract_model_code,
    result: ImageInferenceResponse,
    model_result: ModelDetectionResult,
) -> InternalModelResultResponse:
    expected_internal_code = (
        CONTRACT_TO_INTERNAL_MODEL_CODE[
            contract_model_code
        ]
    )

    if model_result.model_code is not expected_internal_code:
        raise RuntimeError(
            "Inference model result order does not match "
            "the requested model order."
        )

    detections = [
        InternalDetectionResponse(
            detection_index=detection_index,
            class_index=detection.class_id,
            raw_name=detection.raw_name,
            class_code=detection.class_code,
            incident_target=detection.incident_target,
            confidence=detection.confidence,
            bounding_box=_create_bounding_box_response(
                result,
                detection,
            ),
        )
        for detection_index, detection in enumerate(
            model_result.detections
        )
    ]

    return InternalModelResultResponse(
        model_code=contract_model_code,
        detection_count=len(detections),
        detections=detections,
    )


def _create_response(
    *,
    payload: InternalInferenceRequest,
    context: InternalRequestContext,
    result: ImageInferenceResponse,
    processing_time_ms: int,
) -> InternalInferenceResponse:
    if len(payload.execution.model_codes) != len(
        result.model_results
    ):
        raise RuntimeError(
            "Inference result count does not match "
            "the requested model count."
        )

    model_results = [
        _create_model_result_response(
            contract_model_code=contract_model_code,
            result=result,
            model_result=model_result,
        )
        for contract_model_code, model_result in zip(
            payload.execution.model_codes,
            result.model_results,
            strict=True,
        )
    ]

    return InternalInferenceResponse(
        data=InternalInferenceData(
            request_id=payload.request_id,
            inference_run_public_id=(
                payload.inference_run_public_id
            ),
            video_frame_public_id=(
                payload.video_frame.public_id
            ),
            image=InternalImageResponse(
                width=result.image_width,
                height=result.image_height,
            ),
            model_count=len(model_results),
            total_detection_count=(
                result.total_detection_count
            ),
            incident_detection_count=(
                result.incident_detection_count
            ),
            processing_time_ms=processing_time_ms,
            model_results=model_results,
        ),
        trace_id=context.trace_id,
    )


@router.post(
    "/inferences",
    response_model=InternalInferenceResponse,
)
async def run_internal_inference(
    payload: InternalInferenceRequest,
    request: Request,
    context: Annotated[
        InternalRequestContext,
        Depends(require_internal_request),
    ],
) -> InternalInferenceResponse:
    validate_request_id_matches_body(
        context,
        payload,
    )

    model_registry: ModelRegistry | None = getattr(
        request.app.state,
        "model_registry",
        None,
    )

    if model_registry is None:
        raise InternalAPIError(
            status_code=503,
            code="AI_INFERENCE_UNAVAILABLE",
            message="AI 추론 서비스를 사용할 수 없습니다.",
            trace_id=context.trace_id,
        )

    started_at = perf_counter()

    try:
        image_bytes = await run_in_threadpool(
            image_download_service.download,
            str(payload.input.download_url),
            expected_content_type=payload.input.mime_type,
        )
    except ImageDownloadInputError as error:
        raise InternalAPIError(
            status_code=400,
            code="AI_INPUT_INVALID",
            message="입력 이미지 요청이 올바르지 않습니다.",
            trace_id=context.trace_id,
            details={
                "reason": str(error),
            },
        ) from error
    except ImageDownloadUnavailableError as error:
        raise InternalAPIError(
            status_code=502,
            code="AI_INPUT_DOWNLOAD_FAILED",
            message="입력 이미지 다운로드에 실패했습니다.",
            trace_id=context.trace_id,
        ) from error

    try:
        result = await run_in_threadpool(
            detection_service.predict,
            image_bytes,
            model_registry,
            model_codes=(
                payload.execution.get_internal_model_codes()
            ),
        )
    except DetectionInputError as error:
        raise InternalAPIError(
            status_code=400,
            code="AI_INPUT_INVALID",
            message="입력 이미지 요청이 올바르지 않습니다.",
            trace_id=context.trace_id,
            details={
                "reason": str(error),
            },
        ) from error
    except DetectionUnavailableError as error:
        raise InternalAPIError(
            status_code=503,
            code="AI_INFERENCE_UNAVAILABLE",
            message="AI 추론 서비스를 사용할 수 없습니다.",
            trace_id=context.trace_id,
        ) from error
    except DetectionServiceError as error:
        raise InternalAPIError(
            status_code=500,
            code="AI_INFERENCE_FAILED",
            message="AI 추론 처리에 실패했습니다.",
            trace_id=context.trace_id,
        ) from error

    processing_time_ms = max(
        round(
            (perf_counter() - started_at) * 1000
        ),
        0,
    )

    try:
        return _create_response(
            payload=payload,
            context=context,
            result=result,
            processing_time_ms=processing_time_ms,
        )
    except RuntimeError as error:
        raise InternalAPIError(
            status_code=500,
            code="AI_INFERENCE_RESULT_INVALID",
            message="AI 추론 결과 형식이 올바르지 않습니다.",
            trace_id=context.trace_id,
        ) from error
