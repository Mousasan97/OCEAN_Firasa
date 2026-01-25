"""
Health check and system info routes
"""
from fastapi import APIRouter, Depends
from datetime import datetime

from src.api.schemas.response import HealthResponse, TraitDescriptionsResponse
from src.services.model_manager import ModelManager, get_model_manager
from src.services.prediction_service import PredictionService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Health check endpoint
    Returns API status and model information
    """
    model_info = model_manager.get_model_info() if model_manager.is_loaded else None

    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        version=settings.APP_VERSION,
        model_loaded=model_manager.is_loaded,
        model_info=model_info,
        timestamp=datetime.utcnow()
    )


@router.get("/ready")
async def readiness_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Readiness check for Kubernetes/orchestration
    Returns 200 if ready to serve, 503 if not
    """
    if model_manager.is_loaded:
        return {"status": "ready"}
    else:
        from fastapi import status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "reason": "Model not loaded"}
        )


@router.get("/traits", response_model=TraitDescriptionsResponse)
async def get_trait_descriptions(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Get personality trait descriptions
    """
    from src.services.prediction_service import PredictionService

    # Create prediction service
    prediction_service = PredictionService(model_manager)

    # Get trait descriptions
    descriptions = prediction_service.get_trait_descriptions()

    return TraitDescriptionsResponse(traits=descriptions)


@router.get("/info")
async def get_api_info():
    """Get API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }
