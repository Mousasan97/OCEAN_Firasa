"""
FastAPI dependencies
"""
from functools import lru_cache

from src.services.model_manager import ModelManager, get_model_manager
from src.services.prediction_service import PredictionService
from src.services.preprocessing_service import PreprocessingService
from src.infrastructure.cache import CacheService, get_cache_service
from src.infrastructure.storage import StorageService, get_storage_service
from src.infrastructure.database import DatabaseService, get_database_service


# Model Manager (singleton)
def get_model_manager_dependency() -> ModelManager:
    """Get model manager instance"""
    return get_model_manager()


# Prediction Service
@lru_cache()
def get_prediction_service_cached() -> PredictionService:
    """Get cached prediction service instance"""
    model_manager = get_model_manager()
    preprocessing_service = PreprocessingService()
    return PredictionService(model_manager, preprocessing_service)


def get_prediction_service() -> PredictionService:
    """Get prediction service instance"""
    return get_prediction_service_cached()


# Infrastructure Services
def get_cache_service_dependency() -> CacheService:
    """Get cache service instance"""
    return get_cache_service()


def get_storage_service_dependency() -> StorageService:
    """Get storage service instance"""
    return get_storage_service()


def get_database_service_dependency() -> DatabaseService:
    """Get database service instance"""
    return get_database_service()
