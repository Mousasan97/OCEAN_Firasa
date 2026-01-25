"""
Request/response logging middleware
"""
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses"""

    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}", extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None
        })

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(f"Response: {response.status_code} in {duration:.3f}s", extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_seconds": duration
        })

        # Add duration header
        response.headers["X-Process-Time"] = str(duration)

        return response
