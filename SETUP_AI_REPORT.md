# Setup Guide: AI Personality Report Feature

## Quick Start

Your configuration has been updated to support AI-generated personality reports. Follow these steps:

### 1. Install Required Dependencies

```bash
# Make sure you're in your virtual environment
# venv311\Scripts\activate (Windows)

# Install AI agent dependencies
pip install anthropic-agents openai scipy
```

Or use the provided requirements file:

```bash
pip install -r requirements_ai_agent.txt
```

### 2. Configuration

The `.env` file already contains `OPENAI_API_KEY`. The system will now accept this configuration.

**New configuration options in `.env`:**

```env
# AI Agent Configuration (already auto-configured)
OPENAI_API_KEY=sk-proj-...  # Already present
ANTHROPIC_API_KEY=  # Optional, if using Anthropic models
AI_REPORT_ENABLED=true  # Default: true
AI_REPORT_MODEL=gpt-5  # Default: gpt-5
AI_REPORT_REASONING_EFFORT=medium  # Options: low, medium, high
```

### 3. Start the Server

```bash
python run_server.py
```

The server should now start without the configuration error!

### 4. Test the Feature

**Option A: Via Web Interface**

1. Open http://localhost:8000
2. Upload an image
3. Click "Analyze Person"
4. See the AI report displayed at the top in a beautiful purple gradient card!

**Option B: Via API**

```bash
curl -X POST "http://localhost:8000/api/v1/predict/upload?generate_report=true&include_interpretations=true" \
  -F "file=@data/image.jpg"
```

**Option C: Via Python Test Script**

```bash
python test_interpretation.py
```

(Modify the script to add `generate_report=true` parameter)

## What Changed

### ✅ Configuration Updates

**File:** `src/utils/config.py`

Added new fields to the `Settings` class:

```python
# AI Agent (for personality report generation)
OPENAI_API_KEY: Optional[str] = None
ANTHROPIC_API_KEY: Optional[str] = None
AI_REPORT_ENABLED: bool = True
AI_REPORT_MODEL: str = "gpt-5"
AI_REPORT_REASONING_EFFORT: str = "medium"
```

This fixes the validation error: `Extra inputs are not permitted`

### ✅ Report Service Updates

**File:** `src/services/report_service.py`

- Now reads configuration from `settings`
- Checks if AI report is enabled before initializing
- Uses configurable model and reasoning effort
- Graceful error handling when disabled

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'anthropic-agents'"

**Solution:**
```bash
pip install anthropic-agents openai
```

### Error: "AI report generation is disabled or not configured"

**Solution:**
Make sure `AI_REPORT_ENABLED=true` in `.env` file

### Error: "Extra inputs are not permitted [OPENAI_API_KEY]"

**Solution:**
This is now fixed! The configuration accepts OPENAI_API_KEY.
Restart the server to load the updated configuration.

### Error: API rate limits or authentication

**Solution:**
Check your OPENAI_API_KEY is valid and has credits/quota.

## Feature Flags

You can control the AI report feature via environment variables:

### Disable AI Reports

```env
AI_REPORT_ENABLED=false
```

When disabled:
- The `generate_report` parameter will be ignored
- No AI calls will be made
- Response will not include `narrative_report` field
- Other features (predictions, interpretations) work normally

### Change AI Model

```env
AI_REPORT_MODEL=gpt-4  # Or claude-3-5-sonnet, etc.
```

### Adjust Reasoning Effort

```env
AI_REPORT_REASONING_EFFORT=low  # Faster, less thorough
AI_REPORT_REASONING_EFFORT=high  # Slower, more thorough
```

## API Usage Examples

### Request with AI Report

```bash
POST /api/v1/predict/upload?generate_report=true&include_interpretations=true
Content-Type: multipart/form-data

file: <image.jpg>
```

### Request without AI Report (faster)

```bash
POST /api/v1/predict/upload?include_interpretations=true
# generate_report defaults to false
```

### Python Example

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/upload',
        files={'file': f},
        params={
            'use_tta': True,
            'method': 'face+middle',
            'include_interpretations': True,
            'generate_report': True  # Enable AI report
        }
    )

result = response.json()

# Access the AI-generated report
if 'narrative_report' in result:
    print("AI Report:")
    print(result['narrative_report'])
```

## Performance Notes

- **With AI report**: Response time ~5-10 seconds (includes AI generation)
- **Without AI report**: Response time ~1-3 seconds (normal prediction)
- **Caching**: Reports are cached, subsequent requests are fast

## Cost Considerations

Each AI report generation calls the OpenAI API:

- **Model**: GPT-5 (or configured model)
- **Tokens**: ~500-1000 tokens per request
- **Cost**: ~$0.01-0.05 per report (varies by model)

Consider:
- Using `generate_report=false` by default
- Only enabling for specific use cases
- Implementing usage quotas for production

## Next Steps

1. ✅ Server starts without errors
2. ✅ Configuration accepts AI keys
3. Test the feature with real data
4. Review AI-generated reports for quality
5. Customize agent instructions if needed (in `report_service.py`)
6. Set up monitoring for API costs

---

**Need help?** Check [AI_REPORT_FEATURE.md](AI_REPORT_FEATURE.md) for complete documentation.
