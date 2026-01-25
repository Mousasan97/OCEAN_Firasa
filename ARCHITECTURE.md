# OCEAN API - Production Architecture

## Overview

This document describes the production-ready architecture for the OCEAN Personality Prediction API, refactored from a single monolithic script into a scalable, maintainable, modular application.

## Architecture Pattern

**Clean Architecture (Layered Architecture)**

The application is organized into distinct layers with clear dependencies flowing inward:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                      │
│  • Routes (health, predict, batch)                          │
│  • Middleware (logging, rate limiting, error handling)      │
│  • Schemas (Pydantic request/response models)               │
└────────────────────┬────────────────────────────────────────┘
                     │ depends on ↓
┌────────────────────▼────────────────────────────────────────┐
│                    Service Layer                             │
│  • PredictionService (orchestration)                        │
│  • PreprocessingService (image preprocessing)              │
│  • ModelManager (singleton, model lifecycle)               │
└────────────────────┬────────────────────────────────────────┘
                     │ depends on ↓
┌────────────────────▼────────────────────────────────────────┐
│                     Core Layer                               │
│  • PersonalityPredictor (ML inference)                      │
│  • FaceDetector (face detection logic)                      │
│  • VideoProcessor (video frame extraction)                  │
│  • ResNetRegressor (model definition)                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                          │
│  • CacheService (Redis/Memory caching)                      │
│  • StorageService (Local/S3 file storage)                   │
│  • DatabaseService (PostgreSQL - optional)                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Utilities                                 │
│  • Configuration (environment-based)                         │
│  • Logging (structured JSON logging)                        │
│  • Exceptions (custom exception hierarchy)                  │
└─────────────────────────────────────────────────────────────┘
```

## Layer Responsibilities

### 1. API Layer (`src/api/`)

**Purpose**: HTTP interface and request handling

**Components**:
- **Routes** (`routes/`): REST endpoint definitions
  - `health.py`: Health checks, readiness probes, system info
  - `predict.py`: Single image/video prediction
  - `batch.py`: Batch prediction for multiple images

- **Middleware** (`middleware/`):
  - `logging.py`: Request/response logging with timing
  - `rate_limit.py`: Simple in-memory rate limiting
  - `error_handler.py`: Global exception handling

- **Schemas** (`schemas/`):
  - `request.py`: Pydantic request validation models
  - `response.py`: Pydantic response models

- **Dependencies** (`dependencies.py`): Dependency injection helpers

**Key Features**:
- OpenAPI/Swagger automatic documentation
- Request validation with Pydantic
- CORS support
- Rate limiting
- Structured error responses

### 2. Service Layer (`src/services/`)

**Purpose**: Business logic orchestration and coordination

**Components**:

- **ModelManager** (`model_manager.py`):
  - Singleton pattern for model lifecycle management
  - Loads model once on startup, caches in memory
  - Handles model warmup and cleanup
  - Thread-safe model access

- **PredictionService** (`prediction_service.py`):
  - Orchestrates full prediction pipeline
  - Handles file uploads, preprocessing, inference
  - Integrates with caching layer
  - Supports batch predictions

- **PreprocessingService** (`preprocessing_service.py`):
  - Image/video preprocessing
  - Face detection and cropping
  - Format conversions (bytes → PIL → numpy)
  - Multiple preprocessing strategies

**Design Patterns**:
- Singleton (ModelManager)
- Facade (PredictionService)
- Strategy (preprocessing methods)

### 3. Core Layer (`src/core/`)

**Purpose**: Pure domain logic, no external dependencies

**Components**:

- **PersonalityPredictor** (`personality_predictor.py`):
  - Core ML inference logic
  - Test Time Augmentation (TTA)
  - Batch inference support
  - Model-agnostic interface

- **FaceDetector** (`face_detector.py`):
  - Haar Cascade face detection
  - Bounding box expansion
  - Face cropping logic

- **VideoProcessor** (`video_processor.py`):
  - Frame extraction (middle, specific position, multiple)
  - Video metadata retrieval

- **Models** (`models/`):
  - `resnet_regressor.py`: ResNet-based regression model

**Design Principles**:
- No framework dependencies
- Pure functions where possible
- Easily testable
- Domain-driven design

### 4. Infrastructure Layer (`src/infrastructure/`)

**Purpose**: External system integrations

**Components**:

- **CacheService** (`cache.py`):
  - Abstract cache backend interface
  - Memory cache (development)
  - Redis cache (production)
  - Automatic key generation

- **StorageService** (`storage.py`):
  - Abstract storage backend interface
  - Local filesystem storage
  - AWS S3 storage
  - Secure path validation

- **DatabaseService** (`database.py`):
  - Optional database integration
  - Placeholder for SQLAlchemy
  - Prediction history tracking

**Design Patterns**:
- Strategy (swappable backends)
- Adapter (unified interface)

### 5. Utilities (`src/utils/`)

**Cross-cutting concerns**:

- **Configuration** (`config.py`):
  - Environment-based settings with Pydantic
  - Type-safe configuration
  - Defaults for all environments
  - Cached settings instance

- **Logging** (`logger.py`):
  - Structured JSON logging (production)
  - Colored console logging (development)
  - Contextual information
  - Log level management

- **Exceptions** (`exceptions.py`):
  - Custom exception hierarchy
  - HTTP status code mapping
  - Detailed error information

## Data Flow

### Single Prediction Request

```
1. Client uploads image
   ↓
2. FastAPI route validates request (Pydantic schema)
   ↓
3. Rate limit middleware checks request count
   ↓
4. PredictionService checks cache for existing result
   ↓ (cache miss)
5. PreprocessingService loads and preprocesses image
   ├─→ FaceDetector detects and crops face (if enabled)
   └─→ Converts to PIL Image
   ↓
6. PersonalityPredictor runs inference
   ├─→ Applies image transforms
   ├─→ Runs model forward pass
   └─→ Applies TTA (if enabled)
   ↓
7. PredictionService caches result
   ↓
8. Response formatted (Pydantic schema)
   ↓
9. Logging middleware logs request/response
   ↓
10. JSON response returned to client
```

### Startup Sequence

```
1. Load environment variables (.env)
   ↓
2. Initialize logging system
   ↓
3. FastAPI lifespan startup
   ↓
4. ModelManager loads checkpoint
   ├─→ Validate checkpoint exists
   ├─→ Load model architecture
   ├─→ Load trained weights
   ├─→ Move to device (CPU/GPU)
   └─→ Run warmup inference
   ↓
5. Initialize infrastructure services
   ├─→ Connect to Redis (if enabled)
   ├─→ Initialize storage backend
   └─→ Connect to database (if enabled)
   ↓
6. API ready to serve requests
```

## Key Design Decisions

### 1. Singleton Pattern for Model

**Why**: Models are large (100MB+) and expensive to load

**Benefits**:
- Load once on startup
- Shared across all requests
- Fast inference (model stays in GPU memory)
- Reduced memory footprint

### 2. Dependency Injection

**Why**: Testability and flexibility

**Implementation**: FastAPI's `Depends()` mechanism

**Benefits**:
- Easy to mock in tests
- Swappable implementations
- Clear dependency graph

### 3. Layered Architecture

**Why**: Separation of concerns, maintainability

**Benefits**:
- Core logic independent of framework
- Easy to swap FastAPI for another framework
- Clear boundaries and responsibilities
- Testable in isolation

### 4. Abstract Infrastructure

**Why**: Support multiple backends (Redis/Memory, S3/Local)

**Benefits**:
- Development without external services
- Production-ready with Redis/S3
- Easy to add new backends
- Consistent interface

### 5. Structured Logging

**Why**: Production observability

**Benefits**:
- Machine-parseable logs (JSON)
- Easy integration with log aggregators
- Rich contextual information
- Performance tracking

## Scalability Considerations

### Horizontal Scaling

**Current**: Single container, multiple workers

**Future**: Multiple containers behind load balancer

```
                    ┌─────────────┐
                    │ Load Balancer│
                    │   (Nginx)    │
                    └──────┬───────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
        ┌───▼───┐      ┌───▼───┐      ┌───▼───┐
        │ API-1 │      │ API-2 │      │ API-3 │
        └───┬───┘      └───┬───┘      └───┬───┘
            │              │              │
            └──────────────┼──────────────┘
                           │
                    ┌──────▼───────┐
                    │    Redis     │
                    │   (Shared)   │
                    └──────────────┘
```

### Performance Optimizations

1. **Model Loading**: Loaded once, shared by all workers
2. **Caching**: Redis for repeated predictions
3. **Batch Processing**: Process multiple images in parallel
4. **TTA Optional**: Skip for faster predictions
5. **Connection Pooling**: Redis connection pool
6. **Async I/O**: FastAPI async endpoints

### Resource Management

- **Memory**: Model stays in RAM, ~500MB-2GB
- **CPU**: Uvicorn workers (1 per core recommended)
- **GPU**: Single model instance, shared inference
- **Disk**: Minimal (logs, temp uploads)

## Deployment Options

### 1. Docker Compose (Current)

**Best for**: Small-medium deployments, development

**Stack**:
- API container (Python + FastAPI)
- Redis container (caching)
- Nginx container (optional, reverse proxy)

### 2. Kubernetes

**Best for**: Large-scale production

**Components**:
- Deployment (API pods with replicas)
- Service (load balancing)
- ConfigMap (configuration)
- Secret (sensitive data)
- PersistentVolume (model storage)
- Redis StatefulSet

### 3. Serverless (Future)

**Best for**: Variable traffic, cost optimization

**Considerations**:
- Cold start latency (model loading)
- Function timeout limits
- Storage for large models

## Security

### Current Implementations

1. **Input Validation**: Pydantic schemas
2. **File Upload Limits**: Max size enforcement
3. **Rate Limiting**: Per-client request limits
4. **Path Traversal Protection**: Storage service validation
5. **CORS**: Configured allowed origins

### Future Enhancements

1. **Authentication**: JWT tokens, API keys
2. **Authorization**: Role-based access control
3. **File Scanning**: Malware detection
4. **HTTPS**: TLS termination at Nginx
5. **Secrets Management**: Vault integration

## Testing Strategy

### Unit Tests
- Core layer functions
- Service layer methods
- Utility functions

### Integration Tests
- API endpoints
- Database operations
- Cache operations

### Performance Tests
- Load testing with locust
- Latency benchmarks
- Memory profiling

## Monitoring & Observability

### Current

- **Logging**: Structured JSON logs
- **Metrics**: Request timing via middleware
- **Health Checks**: `/health`, `/ready` endpoints

### Future

- **APM**: Application Performance Monitoring
- **Tracing**: Distributed tracing (OpenTelemetry)
- **Metrics**: Prometheus + Grafana
- **Alerting**: Alert on errors, high latency
- **Error Tracking**: Sentry integration

## Migration from Original Script

### Before (Monolithic Script)
- Single 413-line file
- All logic intertwined
- Hard to test
- Hard to scale
- No API layer

### After (Production Architecture)
- 30+ modular files
- Clear separation of concerns
- Fully testable
- Horizontally scalable
- REST API with docs

### Breaking Changes
None - CLI script preserved for backward compatibility

## Future Enhancements

1. **Model Versioning**: A/B testing, gradual rollouts
2. **Async Processing**: Celery for video processing
3. **WebSocket Support**: Real-time predictions
4. **GraphQL API**: Alternative to REST
5. **Multi-Model Support**: Ensemble predictions
6. **Confidence Scores**: Uncertainty estimation
7. **Explainability**: SHAP, Grad-CAM visualizations
8. **User Management**: Accounts, usage tracking

## References

- Clean Architecture: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- FastAPI: https://fastapi.tiangolo.com/
- Twelve-Factor App: https://12factor.net/
