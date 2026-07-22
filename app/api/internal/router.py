from fastapi import APIRouter

from app.api.internal import inferences


internal_api_router = APIRouter()
internal_api_router.include_router(
    inferences.router,
    tags=["internal-inferences"],
)
