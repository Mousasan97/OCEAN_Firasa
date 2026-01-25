# Testing the OCEAN API

## Quick Start - 3 Steps

### 1. Start the Server

```bash
# Make sure you're in the OCEAN directory with your virtual env activated
python run_server.py
```

You should see:
```
🚀 Starting OCEAN Personality API Server
✓ Model found: output/single_img_resnet18/best.pt
✓ Configuration: .env

Loading model...
Model loaded successfully
API ready at http://0.0.0.0:8000
```

### 2. Test the Prediction

**Open a NEW terminal** (keep the server running) and run:

```bash
python quick_test.py
```

Expected output:
```
🖼️  Testing prediction with: data/image.jpg
📡 Sending request to API...

======================================================================
✅ PERSONALITY PREDICTION RESULTS
======================================================================

Big-5 Personality Traits:

HIGH     Extraversion: +0.650  (Outgoing, social, energetic)
LOW      Neuroticism: -0.230  (Anxious, moody, emotionally unstable)
MED      Agreeableness: +0.450  (Cooperative, trusting, helpful)
HIGH     Conscientiousness: +0.780  (Organized, disciplined, responsible)
MED      Openness: +0.340  (Creative, curious, open to new experiences)

======================================================================
Face Detected: True
Method Used: face+middle
======================================================================
```

### 3. Explore the Interactive Docs

Open your browser and go to: **http://localhost:8000/docs**

You'll see the **Swagger UI** where you can:
- ✅ See all available endpoints
- ✅ Upload images directly in the browser
- ✅ Test all API features interactively

---

## All Testing Methods

### Method 1: Python Test Script (Easiest)

```bash
# Run comprehensive tests
python test_api.py

# Or just quick test
python quick_test.py
```

### Method 2: Browser Interactive Docs (Most Visual)

1. Open: http://localhost:8000/docs
2. Find "POST /api/v1/predict/upload"
3. Click "Try it out"
4. Upload `data/image.jpg`
5. Click "Execute"
6. See the results!

### Method 3: cURL Commands

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get trait descriptions
curl http://localhost:8000/api/v1/traits

# Predict from image (Windows)
curl -X POST "http://localhost:8000/api/v1/predict/upload" ^
  -F "file=@data/image.jpg" ^
  -F "use_tta=true" ^
  -F "method=face+middle"

# Predict from image (Linux/Mac)
curl -X POST "http://localhost:8000/api/v1/predict/upload" \
  -F "file=@data/image.jpg" \
  -F "use_tta=true" \
  -F "method=face+middle"
```

### Method 4: Python Requests Library

```python
import requests

# Predict
with open('data/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/upload',
        files={'file': f},
        params={'use_tta': True, 'method': 'face+middle'}
    )

print(response.json())
```

### Method 5: Postman/Insomnia

1. Create POST request to: `http://localhost:8000/api/v1/predict/upload`
2. Set body type to `form-data`
3. Add field: `file` (type: File) → select `data/image.jpg`
4. Add field: `use_tta` (type: Text) → value: `true`
5. Add field: `method` (type: Text) → value: `face+middle`
6. Send request

---

## Testing Different Features

### Test 1: Basic Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "model_info": {
    "loaded": true,
    "backbone": "resnet18",
    "device": "cuda",
    "trait_labels": ["extraversion", "neuroticism", ...]
  }
}
```

### Test 2: Prediction with Face Detection

```bash
python quick_test.py
```

### Test 3: Prediction WITHOUT Face Detection

```python
import requests

with open('data/image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/upload',
        files={'file': f},
        params={'use_tta': False, 'method': 'middle'}  # No face detection
    )

print(response.json())
```

### Test 4: Batch Prediction (Multiple Images)

If you have multiple images:

```python
import requests

files = [
    ('files', open('data/image.jpg', 'rb')),
    ('files', open('data/image2.jpg', 'rb')),
]

response = requests.post(
    'http://localhost:8000/api/v1/batch/upload',
    files=files,
    params={'use_tta': True}
)

print(response.json())
```

---

## Troubleshooting

### ❌ Server won't start

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the OCEAN directory
cd "c:\Users\MousaSondoqah-Becque\OneDrive - ICARES\Desktop\Faraseh\OCEAN"

# Try running directly
python -m src.api.main
```

---

### ❌ Model not loading

**Problem**: `FileNotFoundError: Checkpoint not found`

**Solution**: Check your `.env` file:
```bash
MODEL_CHECKPOINT_PATH=output/single_img_resnet18/best.pt
```

Make sure the file exists:
```bash
ls output/single_img_resnet18/best.pt
```

---

### ❌ Cannot connect to API

**Problem**: `Connection refused` when running test script

**Solution**: Make sure the server is running in another terminal:
```bash
python run_server.py
```

---

### ❌ Face detection error

**Problem**: `No face detected in image`

**Solution**: Use `method=face+middle` to fallback to full image:
```python
params={'method': 'face+middle'}  # Will use full image if no face found
```

---

## Performance Benchmarks

### Single Prediction
- **With TTA (more accurate)**: ~200-500ms
- **Without TTA (faster)**: ~100-300ms
- **Cached result**: ~5-10ms

### Batch Prediction (10 images)
- **With TTA**: ~800-1500ms
- **Without TTA**: ~400-800ms

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root info |
| `/docs` | GET | Interactive API docs |
| `/api/v1/health` | GET | Health check |
| `/api/v1/ready` | GET | Readiness probe |
| `/api/v1/traits` | GET | Trait descriptions |
| `/api/v1/info` | GET | API information |
| `/api/v1/predict/upload` | POST | Upload & predict |
| `/api/v1/predict/file` | POST | Predict from server path |
| `/api/v1/predict/demo` | GET | Demo response |
| `/api/v1/batch/upload` | POST | Batch prediction |
| `/api/v1/batch/demo` | GET | Demo batch response |

---

## Next Steps

Once you've tested the API:

1. ✅ **Integrate with frontend**: Use the REST API from your web/mobile app
2. ✅ **Deploy with Docker**: `cd docker && docker-compose up`
3. ✅ **Add monitoring**: Set up logging aggregation
4. ✅ **Scale horizontally**: Add more API containers
5. ✅ **Add authentication**: Implement API keys/JWT

---

## Example Integration

### JavaScript/TypeScript (React, Vue, etc.)

```javascript
async function predictPersonality(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile);

  const response = await fetch(
    'http://localhost:8000/api/v1/predict/upload?use_tta=true&method=face+middle',
    {
      method: 'POST',
      body: formData
    }
  );

  const result = await response.json();
  return result.predictions;
}
```

### Python

```python
import requests

def predict_personality(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:8000/api/v1/predict/upload',
            files={'file': f},
            params={'use_tta': True, 'method': 'face+middle'}
        )
    return response.json()['predictions']
```

---

## Support

If you encounter any issues:
1. Check the server logs in the terminal
2. Visit `/docs` for interactive testing
3. Ensure all dependencies are installed
4. Verify model checkpoint exists

Happy testing! 🎉
