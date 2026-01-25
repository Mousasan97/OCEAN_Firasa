"""
Batch prediction routes
"""
from fastapi import APIRouter, UploadFile, File, Depends, Query, HTTPException, status
from typing import List
from PIL import Image
from io import BytesIO

from src.api.schemas.response import BatchPredictionResponse
from src.services.model_manager import ModelManager, get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/batch", tags=["Batch Prediction"])


async def get_prediction_service(
    model_manager: ModelManager = Depends(get_model_manager)
) -> PredictionService:
    """Dependency to get prediction service"""
    return PredictionService(
        model_manager=model_manager,
        preprocessing_service=PreprocessingService()
    )


@router.post("/upload", response_model=BatchPredictionResponse)
async def batch_predict_from_uploads(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    use_tta: bool = Query(default=True, description="Use Test Time Augmentation"),
    method: str = Query(default="face+middle", description="Preprocessing method"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Batch predict personality from multiple uploaded images

    Upload multiple images and get predictions for all of them in a single request.
    This is more efficient than making individual requests for each image.

    **Maximum files:** Limited by server configuration
    **Supported formats:** Same as single prediction endpoint

    **Note:** All images are processed with the same preprocessing method and TTA setting.
    """
    logger.info(f"Batch prediction request: {len(files)} files")

    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    if len(files) > 50:  # Configurable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many files (maximum 50)"
        )

    try:
        preprocessing_service = PreprocessingService()
        processed_images = []
        filenames = []

        # Preprocess all images
        for file in files:
            logger.info(f"Processing file: {file.filename}")

            # Read file
            content = await file.read()

            # Check size
            if len(content) > settings.MAX_UPLOAD_SIZE:
                logger.warning(f"Skipping {file.filename}: file too large")
                continue

            # Preprocess
            try:
                processed_image, metadata = preprocessing_service.preprocess_from_bytes(
                    content,
                    method=method
                )
                processed_images.append(processed_image)
                filenames.append(file.filename)
            except Exception as e:
                logger.warning(f"Failed to preprocess {file.filename}: {e}")
                continue

        if len(processed_images) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid images could be processed"
            )

        # Batch predict
        logger.info(f"Running batch prediction for {len(processed_images)} images...")
        results = prediction_service.predict_batch(
            pil_images=processed_images,
            use_tta=use_tta
        )

        # Add filenames to results
        for i, result in enumerate(results):
            if i < len(filenames):
                result["metadata"]["filename"] = filenames[i]

        # Format response
        predictions = [result["predictions"] for result in results]

        return BatchPredictionResponse(
            success=True,
            predictions=predictions,
            count=len(predictions),
            metadata={
                "total_uploaded": len(files),
                "total_processed": len(processed_images),
                "use_tta": use_tta,
                "method": method,
                "filenames": filenames
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/demo")
async def demo_batch_prediction():
    """
    Demo endpoint showing example batch prediction response
    """
    return {
        "success": True,
        "predictions": [
            {
                "extraversion": 0.65,
                "neuroticism": -0.23,
                "agreeableness": 0.45,
                "conscientiousness": 0.78,
                "openness": 0.34
            },
            {
                "extraversion": 0.12,
                "neuroticism": 0.56,
                "agreeableness": 0.78,
                "conscientiousness": -0.12,
                "openness": 0.89
            }
        ],
        "count": 2,
        "metadata": {
            "total_uploaded": 2,
            "total_processed": 2,
            "use_tta": True,
            "method": "face+middle"
        },
        "note": "This is a demo response. Use POST /batch/upload to get real predictions."
    }
