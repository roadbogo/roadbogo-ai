from fastapi import APIRouter

from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services.inference_service import InferenceService

router = APIRouter()
service = InferenceService()


@router.post("/", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest) -> InferenceResponse:
    return service.predict(payload)
