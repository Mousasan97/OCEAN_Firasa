# OCEAN Personality Prediction API

Production-ready REST API for predicting Big-5 personality traits from facial images using deep learning.

## Features

- **REST API**: FastAPI-based with automatic OpenAPI documentation
- **Face Detection**: Automatic face detection and cropping
- **Batch Processing**: Process multiple images in parallel
- **Video Support**: Extract frames from videos for analysis
- **Caching**: Redis-based caching for improved performance
- **Production Ready**: Docker, logging, monitoring, rate limiting
- **Clean Architecture**: Modular design with clear separation of concerns

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd OCEAN
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Especially set MODEL_CHECKPOINT_PATH to your trained model
```

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt

# For development
pip install -r requirements/dev.txt
```

### 4. Run API Server

```bash
# Development mode (auto-reload)
python -m src.api.main

# Or using uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
cd docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services included:
- **api**: Main FastAPI application
- **redis**: Cache backend
- **nginx**: Reverse proxy (optional, use with `--profile with-nginx`)

### Using Docker Only

```bash
# Build image
docker build -f docker/Dockerfile -t ocean-api .

# Run container
docker run -p 8000:8000 -v ./output:/app/output ocean-api
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Single Prediction

```bash
# Upload image for prediction
curl -X POST "http://localhost:8000/api/v1/predict/upload" \
  -F "file=@image.jpg" \
  -F "use_tta=true" \
  -F "method=face+middle"
```

Python example:

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/upload',
        files={'file': f},
        params={'use_tta': True, 'method': 'face+middle'}
    )

result = response.json()
print(result['predictions'])
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/batch/upload" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "use_tta=true"
```

## Project Structure

```
OCEAN/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── routes/            # API endpoints
│   │   ├── middleware/        # Custom middleware
│   │   └── schemas/           # Request/response models
│   ├── core/                  # Core domain logic
│   │   ├── models/            # ML model definitions
│   │   ├── face_detector.py   # Face detection
│   │   ├── video_processor.py # Video processing
│   │   └── personality_predictor.py
│   ├── services/              # Business logic
│   │   ├── model_manager.py   # Model loading/caching
│   │   ├── prediction_service.py
│   │   └── preprocessing_service.py
│   ├── infrastructure/        # External services
│   │   ├── cache.py          # Redis/memory cache
│   │   ├── storage.py        # Local/S3 storage
│   │   └── database.py       # Database (optional)
│   └── utils/                # Utilities
│       ├── config.py         # Configuration
│       ├── logger.py         # Logging
│       └── exceptions.py     # Custom exceptions
├── docker/                    # Docker configuration
├── requirements/              # Dependencies
├── config/                    # Environment configs
└── tests/                     # Tests
```

## Architecture

This application follows **Clean Architecture** principles:

1. **API Layer**: FastAPI routes, middleware, schemas
2. **Service Layer**: Business logic orchestration
3. **Core Layer**: Domain logic (face detection, prediction)
4. **Infrastructure Layer**: External dependencies (cache, storage)

### Key Components

- **Model Manager**: Singleton pattern, loads model once on startup
- **Prediction Service**: Orchestrates preprocessing → prediction pipeline
- **Cache Service**: Redis/memory caching for repeated predictions
- **Storage Service**: Local/S3 file storage
- **Preprocessing Service**: Face detection and image preprocessing

## Configuration

All configuration is managed through environment variables (see `.env.example`):

- **Model Settings**: Checkpoint path, device, image size
- **API Settings**: Host, port, CORS, rate limiting
- **Cache**: Backend (memory/redis), TTL
- **Storage**: Backend (local/S3)
- **Logging**: Level, format, output

## API Endpoints

### Health & Info
- `GET /api/v1/health` - Health check
- `GET /api/v1/ready` - Readiness probe
- `GET /api/v1/traits` - Personality trait descriptions
- `GET /api/v1/info` - API information

### Prediction
- `POST /api/v1/predict/upload` - Predict from uploaded file
- `POST /api/v1/predict/file` - Predict from server file path
- `GET /api/v1/predict/demo` - Demo response

### Batch
- `POST /api/v1/batch/upload` - Batch predict from uploads
- `GET /api/v1/batch/demo` - Demo batch response

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint
flake8 src/
```

### Adding New Features

1. Add core logic in `src/core/`
2. Add service layer in `src/services/`
3. Add API route in `src/api/routes/`
4. Update schemas in `src/api/schemas/`
5. Write tests in `tests/`

## Monitoring & Logging

### Logs

All logs use structured JSON format in production:

```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "src.services.prediction",
  "message": "Prediction completed",
  "environment": "production"
}
```

### Metrics

- Request/response times via middleware
- Model inference latency
- Cache hit/miss rates
- Error rates

## Performance

### Optimization Tips

1. **Use Redis caching** for repeated predictions
2. **Enable TTA** only when accuracy is critical (slower)
3. **Batch processing** for multiple images
4. **GPU acceleration** if available (set `MODEL_DEVICE=cuda`)
5. **Adjust workers** based on CPU cores

### Benchmarks

- Single prediction: ~100-300ms (CPU), ~50-100ms (GPU)
- Batch (10 images): ~500-1000ms (CPU), ~200-400ms (GPU)
- With cache hit: ~5-10ms

## Troubleshooting

### Model Not Loading

```bash
# Check model path
ls -l output/single_img_resnet18/best.pt

# Check logs
docker-compose logs api
```

### Out of Memory

```bash
# Reduce workers
export WORKERS=2

# Use CPU instead of GPU
export MODEL_DEVICE=cpu
```

### Redis Connection Failed

```bash
# Use memory cache instead
export CACHE_BACKEND=memory
```

## Production Deployment

### Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure Redis for caching
- [ ] Set up proper logging (JSON format)
- [ ] Enable rate limiting
- [ ] Configure CORS origins
- [ ] Set up monitoring (Sentry, Prometheus)
- [ ] Use HTTPS (configure nginx)
- [ ] Set resource limits (CPU, memory)
- [ ] Configure health checks

### Kubernetes Deployment

Example deployment configuration available in `k8s/` (to be added).

## License

[Your License]

## Citation

If you use this API in research, please cite:

```
[Your Citation]
```

## Important Disclaimer

⚠️ These predictions are based on facial appearance alone and should be
interpreted with caution. Personality is complex and cannot be accurately
determined from photos alone. This tool is intended for research and
educational purposes only.

## Support

For issues and questions:
- GitHub Issues: [Your Repo]
- Email: [Your Email]
- Documentation: http://localhost:8000/docs
