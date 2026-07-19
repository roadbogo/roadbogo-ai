from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.inference.model_registry import ModelRegistry
from app.schemas.detection import ImageInferenceResponse
from app.services.detection_service import (
    MAX_IMAGE_BYTES,
    DetectionInputError,
    DetectionService,
    DetectionServiceError,
    DetectionUnavailableError,
)


router = APIRouter()
service = DetectionService()

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}


@router.post(
    "/",
    response_model=ImageInferenceResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid image data",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "description": "Image file is too large",
        },
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {
            "description": "Unsupported image format",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Inference models are not ready",
        },
    },
)
async def run_inference(
    request: Request,
    file: UploadFile = File(...),
) -> ImageInferenceResponse:
    if file.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        await file.close()

        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=(
                "Only JPEG and PNG image files are supported."
            ),
        )

    model_registry: ModelRegistry | None = getattr(
        request.app.state,
        "model_registry",
        None,
    )

    if (
        model_registry is None
        or not model_registry.is_loaded
    ):
        await file.close()

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Inference models are not ready.",
        )

    try:
        image_bytes = await file.read(
            MAX_IMAGE_BYTES + 1
        )
    finally:
        await file.close()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Uploaded image exceeds the maximum size of "
                f"{MAX_IMAGE_BYTES} bytes."
            ),
        )

    try:
        return await run_in_threadpool(
            service.predict,
            image_bytes,
            model_registry,
        )
    except DetectionInputError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except DetectionUnavailableError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error
    except DetectionServiceError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Inference execution failed.",
        ) from error
