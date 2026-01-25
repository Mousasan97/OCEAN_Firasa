"""
Configuration management with environment variable support
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation"""

    # Application
    APP_NAME: str = "OCEAN Personality API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]

    # Security
    API_KEY_ENABLED: bool = False
    API_KEY: Optional[str] = None
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds

    # Model Type Selection
    # "vat" = Video Attention Transformer (video input, 32 frames, 224x224)
    # "resnet" = ResNet18/50 (single image input, face detection, 256x256)
    MODEL_TYPE: str = "resnet"  # vat or resnet

    # Model Configuration (auto-adjusted based on MODEL_TYPE)
    MODEL_CHECKPOINT_PATH: str = "output/single_img_resnet18/best.pt"  # ResNet default
    MODEL_BACKBONE: str = "resnet18"  # resnet18, resnet50, or vat
    MODEL_IMAGE_SIZE: int = 256  # 256 for ResNet, 224 for VAT
    MODEL_NUM_FRAMES: int = 32  # Number of frames for VAT model (ignored for ResNet)
    MODEL_DEVICE: str = "auto"  # auto, cpu, cuda
    MODEL_WARMUP: bool = True

    # VAT-specific paths (when MODEL_TYPE=vat)
    VAT_CHECKPOINT_PATH: str = "video_model/best_model.pth"

    # ResNet multi-frame settings (when MODEL_TYPE=resnet)
    # Extract multiple frames and average predictions for better accuracy
    RESNET_NUM_FRAMES: int = 10  # Number of frames to extract for ResNet (1 = single frame mode)

    # Preprocessing
    # For ResNet: face, middle, or face+middle (recommended: face+middle)
    # For VAT: none (VAT trained on full frames)
    FACE_DETECTION_METHOD: str = "face+middle"
    FACE_EXPAND_RATIO: float = 0.35
    USE_TTA: bool = True  # Test Time Augmentation

    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"

    # Video Compression (auto-compress videos exceeding threshold)
    VIDEO_COMPRESSION_ENABLED: bool = True
    VIDEO_COMPRESSION_THRESHOLD: int = 30 * 1024 * 1024  # Compress if > 30MB
    VIDEO_COMPRESSION_TARGET_BITRATE: str = "1M"  # Target bitrate (e.g., "1M", "2M", "500k")
    VIDEO_COMPRESSION_MAX_RESOLUTION: int = 720  # Max height (e.g., 720p, 480p)
    VIDEO_COMPRESSION_AUDIO_BITRATE: str = "128k"  # Audio bitrate

    # Cache
    CACHE_ENABLED: bool = True
    CACHE_BACKEND: str = "memory"  # memory, redis
    CACHE_TTL: int = 3600  # seconds
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Database
    DATABASE_ENABLED: bool = False
    DATABASE_URL: Optional[str] = None
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Storage
    STORAGE_BACKEND: str = "local"  # local, s3
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json, text
    LOG_FILE: Optional[str] = None

    # Monitoring
    METRICS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None

    # Background Tasks
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    # AI Agent (for personality report generation)
    AI_REPORT_ENABLED: bool = True
    AI_PROVIDER: str = "openai"  # openai, vertex, gemini, anthropic

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4.1-mini"  # gpt-4.1-mini, gpt-4o, gpt-4o-mini, o1, etc.
    OPENAI_REASONING_EFFORT: str = "medium"  # low, medium, high (for o1 models)

    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # Google Gemini API Configuration (via Google AI Studio API key)
    # Supports both Gemini and Gemma models
    # Note: Gemma models don't support function calling - will use JSON text fallback
    GOOGLE_API_KEY: Optional[str] = None  # Get from https://aistudio.google.com/apikey
    GEMINI_MODEL: str = "gemini-2.5-flash"  # gemini-2.5-flash, gemini-2.5-pro, gemma-3-27b-it (no func calling)

    # Google Vertex AI Configuration (enterprise, uses service account)
    VERTEX_PROJECT: Optional[str] = None
    VERTEX_LOCATION: str = "us-central1"
    VERTEX_MODEL: str = "gemini-2.5-flash"  # gemini-2.5-flash, gemini-2.5-pro, etc.
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None  # Path to service account JSON

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def processed_path(self) -> Path:
        path = Path(self.PROCESSED_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_device_resolved(self) -> str:
        """Resolve auto device to actual device"""
        if self.MODEL_DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.MODEL_DEVICE

    @property
    def is_vat_model(self) -> bool:
        """Check if using VAT model"""
        return self.MODEL_TYPE.lower() == "vat"

    @property
    def is_resnet_model(self) -> bool:
        """Check if using ResNet model"""
        return self.MODEL_TYPE.lower() == "resnet"

    @property
    def effective_checkpoint_path(self) -> str:
        """Get the checkpoint path based on model type"""
        if self.is_vat_model:
            return self.VAT_CHECKPOINT_PATH
        return self.MODEL_CHECKPOINT_PATH

    @property
    def effective_image_size(self) -> int:
        """Get image size based on model type (256 for ResNet, 224 for VAT)"""
        if self.is_vat_model:
            return 224
        return 256

    @property
    def effective_face_detection(self) -> str:
        """Get face detection method based on model type"""
        if self.is_vat_model:
            return "none"  # VAT trained on full frames
        return self.FACE_DETECTION_METHOD


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience accessors
settings = get_settings()
