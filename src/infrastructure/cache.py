"""
Caching layer for predictions
Supports in-memory and Redis backends
"""
import json
import hashlib
from typing import Optional, Any
from abc import ABC, abstractmethod

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import CacheError

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend (simple dict-based)"""

    def __init__(self):
        self._cache = {}
        logger.info("MemoryCache initialized")

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache (TTL not implemented for simplicity)"""
        self._cache[key] = value

    def delete(self, key: str) -> None:
        """Delete value from memory cache"""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear memory cache"""
        self._cache.clear()
        logger.info("Memory cache cleared")

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._cache


class RedisCache(CacheBackend):
    """Redis cache backend"""

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache

        Args:
            redis_url: Redis connection URL (uses config if None)
        """
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required for RedisCache. Install with: pip install redis")

        # Build Redis URL
        if redis_url is None:
            if settings.REDIS_URL:
                redis_url = settings.REDIS_URL
            else:
                redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
                if settings.REDIS_PASSWORD:
                    redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"RedisCache initialized: {redis_url}")
        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {str(e)}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.client.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            raise CacheError(f"Failed to get from cache: {str(e)}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis with TTL"""
        try:
            ttl = ttl or settings.CACHE_TTL
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            raise CacheError(f"Failed to set in cache: {str(e)}")

    def delete(self, key: str) -> None:
        """Delete value from Redis"""
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    def clear(self) -> None:
        """Clear all keys in Redis DB"""
        try:
            self.client.flushdb()
            logger.info("Redis cache cleared")
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False


class CacheService:
    """
    High-level cache service
    Handles prediction result caching with automatic key generation
    """

    def __init__(self, backend: Optional[CacheBackend] = None):
        """
        Initialize cache service

        Args:
            backend: Cache backend (auto-selects based on config if None)
        """
        if backend is None:
            if not settings.CACHE_ENABLED:
                self.backend = None
                logger.info("Caching disabled")
            elif settings.CACHE_BACKEND == "redis":
                self.backend = RedisCache()
            else:
                self.backend = MemoryCache()
        else:
            self.backend = backend

        logger.info(f"CacheService initialized with backend: {type(self.backend).__name__ if self.backend else 'None'}")

    def _generate_key(self, image_hash: str, **kwargs) -> str:
        """
        Generate cache key from image hash and parameters

        Args:
            image_hash: Hash of image data
            **kwargs: Additional parameters (method, tta, etc.)

        Returns:
            Cache key string
        """
        # Sort kwargs for consistent keys
        params = sorted(kwargs.items())
        param_str = "_".join(f"{k}={v}" for k, v in params)

        key = f"prediction:{image_hash}:{param_str}"
        return key

    def get_prediction(self, image_hash: str, **kwargs) -> Optional[dict]:
        """
        Get cached prediction result

        Args:
            image_hash: Hash of image data
            **kwargs: Parameters used for prediction

        Returns:
            Cached prediction result or None
        """
        if self.backend is None:
            return None

        try:
            key = self._generate_key(image_hash, **kwargs)
            result = self.backend.get(key)

            if result is not None:
                logger.info(f"Cache HIT: {key}")
            else:
                logger.debug(f"Cache MISS: {key}")

            return result
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set_prediction(
        self,
        image_hash: str,
        prediction: dict,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Cache prediction result

        Args:
            image_hash: Hash of image data
            prediction: Prediction result to cache
            ttl: Time to live in seconds
            **kwargs: Parameters used for prediction
        """
        if self.backend is None:
            return

        try:
            key = self._generate_key(image_hash, **kwargs)
            self.backend.set(key, prediction, ttl=ttl)
            logger.debug(f"Cached prediction: {key}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def clear(self) -> None:
        """Clear all cache entries"""
        if self.backend is not None:
            self.backend.clear()


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get cache service instance (singleton)"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
