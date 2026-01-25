"""
API request schemas
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Prediction request schema (for JSON-based requests)"""

    use_tta: Optional[bool] = Field(
        default=True,
        description="Whether to use Test Time Augmentation for better accuracy"
    )
    preprocessing_method: Optional[Literal["face", "middle", "face+middle"]] = Field(
        default="face+middle",
        description="Preprocessing method: 'face' (detect face, error if none), 'middle' (use full image), 'face+middle' (try face, fallback to full)"
    )
    save_result: Optional[bool] = Field(
        default=False,
        description="Whether to save prediction result to storage"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_tta": True,
                "preprocessing_method": "face+middle",
                "save_result": False
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction request schema"""

    use_tta: Optional[bool] = Field(default=True)
    preprocessing_method: Optional[Literal["face", "middle", "face+middle"]] = Field(
        default="face+middle"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_tta": True,
                "preprocessing_method": "face+middle"
            }
        }


class FileUploadParams(BaseModel):
    """Query parameters for file upload endpoint"""

    use_tta: bool = Field(
        default=True,
        description="Use Test Time Augmentation"
    )
    method: Literal["face", "middle", "face+middle"] = Field(
        default="face+middle",
        description="Face detection method"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_tta": True,
                "method": "face+middle"
            }
        }
