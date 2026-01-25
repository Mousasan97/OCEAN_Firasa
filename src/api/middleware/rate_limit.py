"""
Simple rate limiting middleware
"""
import time
from collections import defaultdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import RateLimitExceededError

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting
    For production, use Redis-based rate limiting
    """

    def __init__(self, app, requests: int = None, period: int = None):
        """
        Initialize rate limiter

        Args:
            app: FastAPI app
            requests: Max requests per period
            period: Time period in seconds
        """
        super().__init__(app)
        self.requests = requests or settings.RATE_LIMIT_REQUESTS
        self.period = period or settings.RATE_LIMIT_PERIOD
        self.enabled = settings.RATE_LIMIT_ENABLED

        # Store request counts: {client_id: [(timestamp, count), ...]}
        self.request_log = defaultdict(list)

        logger.info(f"RateLimitMiddleware initialized: {self.requests} requests per {self.period}s")

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Use client IP as identifier
        # In production, use API key or user ID
        if request.client:
            return request.client.host
        return "unknown"

    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove old requests outside the time window"""
        cutoff_time = current_time - self.period
        self.request_log[client_id] = [
            (ts, count) for ts, count in self.request_log[client_id]
            if ts > cutoff_time
        ]

    def _check_rate_limit(self, client_id: str) -> tuple[bool, int]:
        """
        Check if client has exceeded rate limit

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        current_time = time.time()

        # Clean old requests
        self._clean_old_requests(client_id, current_time)

        # Count requests in current window
        total_requests = sum(count for _, count in self.request_log[client_id])

        if total_requests >= self.requests:
            # Calculate retry after
            oldest_request = self.request_log[client_id][0][0]
            retry_after = int(self.period - (current_time - oldest_request)) + 1
            return False, retry_after

        # Add current request
        self.request_log[client_id].append((current_time, 1))
        return True, 0

    async def dispatch(self, request: Request, call_next):
        """Check rate limit before processing request"""
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health check
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)

        client_id = self._get_client_id(request)

        # Check rate limit
        is_allowed, retry_after = self._check_rate_limit(client_id)

        if not is_allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise RateLimitExceededError(retry_after=retry_after)

        response = await call_next(request)
        return response
