"""
FastAPI Main Application
Production-ready OCEAN Personality Prediction API
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.routes import health, predict, batch, debug, websocket, stream, chat
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.error_handler import (
    ocean_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    generic_exception_handler
)
from src.services.model_manager import get_model_manager
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import OCEANException

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 70)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info("=" * 70)

    # Load model on startup
    try:
        logger.info("Loading model...")
        model_manager = get_model_manager()
        model_manager.load_model(
            checkpoint_path=settings.MODEL_CHECKPOINT_PATH,
            device=settings.model_device_resolved
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        if settings.is_production:
            raise  # Fail fast in production
        else:
            logger.warning("Continuing without model (development mode)")

    logger.info("=" * 70)
    logger.info(f"API ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"Docs available at http://{settings.HOST}:{settings.PORT}/docs")
    logger.info("=" * 70)

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        model_manager = get_model_manager()
        if model_manager.is_loaded:
            model_manager.unload_model()
            logger.info("Model unloaded")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# OCEAN Personality Prediction API

Predict Big-5 personality traits from facial images using deep learning.

## Features
- **Single Image Prediction**: Upload an image and get personality predictions
- **Batch Prediction**: Process multiple images in one request
- **Video Support**: Extract frames from videos for prediction
- **Face Detection**: Automatic face detection and cropping
- **Caching**: Redis-based caching for faster repeated predictions
- **Production Ready**: Rate limiting, logging, error handling, monitoring

## Personality Traits (Big-5)
- **Extraversion**: Outgoing, social, energetic
- **Neuroticism**: Anxious, moody, emotionally unstable
- **Agreeableness**: Cooperative, trusting, helpful
- **Conscientiousness**: Organized, disciplined, responsible
- **Openness**: Creative, curious, open to new experiences

## API Endpoints
- `POST /api/v1/predict/upload`: Predict from uploaded file
- `POST /api/v1/batch/upload`: Batch predict from multiple files
- `GET /api/v1/health`: Health check
- `GET /api/v1/traits`: Get trait descriptions

## Usage Example
```python
import requests

# Upload image for prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/upload',
        files={'file': f},
        params={'use_tta': True, 'method': 'face+middle'}
    )

result = response.json()
print(result['predictions'])
```

## Important Notice
⚠️ These predictions are based on facial appearance alone and should be
interpreted with caution. Personality is complex and cannot be accurately
determined from photos alone. Use for research and entertainment purposes only.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG
)

# CORS Middleware - Allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else settings.CORS_ORIGINS,
    allow_credentials=False if settings.is_development else settings.CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware
app.add_middleware(LoggingMiddleware)

if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.RATE_LIMIT_REQUESTS,
        period=settings.RATE_LIMIT_PERIOD
    )

# Exception Handlers
app.add_exception_handler(OCEANException, ocean_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include Routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(predict.router, prefix=settings.API_V1_PREFIX)
app.include_router(batch.router, prefix=settings.API_V1_PREFIX)
app.include_router(debug.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)  # WebSocket routes (no prefix needed)
app.include_router(stream.router, prefix=settings.API_V1_PREFIX)  # SSE streaming routes
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)  # AI coach chat routes

# Mount static files for web interface
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"Mounted static files from: {static_path}")

# Root endpoint - serve web interface
@app.get("/")
async def root():
    """Serve web interface"""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Fallback to API info
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health",
            "note": "Web interface not found. Access API docs at /docs"
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        log_level=settings.LOG_LEVEL.lower()
    )
