# AI Personality Report Feature

## Overview

The OCEAN API now includes an AI-powered narrative report generation feature that transforms structured personality data into natural, evidence-based personality reports written in scientific yet human language.

## Architecture

### New Components

1. **Report Service** (`src/services/report_service.py`)
   - Uses Anthropic's Agents SDK
   - Implements `PersonalityReportService` class
   - Singleton pattern for efficient resource usage
   - Async report generation

2. **Updated Prediction Service** (`src/services/prediction_service.py`)
   - New parameter: `generate_report: bool`
   - Two methods:
     - `add_interpretations()` - Synchronous (legacy)
     - `add_interpretations_async()` - Async (supports AI reports)

3. **Updated API Endpoints** (`src/api/routes/predict.py`)
   - Both `/upload` and `/file` endpoints now support `generate_report` parameter
   - Example: `POST /api/v1/predict/upload?generate_report=true`

4. **Updated Response Schema** (`src/api/schemas/response.py`)
   - New field: `narrative_report: Optional[str]`
   - New field: `report_error: Optional[str]`

5. **Updated Frontend** (`static/`)
   - `index.html`: New `narrativeReport` container
   - `app.js`: New `displayNarrativeReport()` function
   - `style.css`: Beautiful gradient card styling for AI reports

## How It Works

### Pipeline Flow

```
1. User uploads image/video
   ↓
2. ML model predicts personality traits (raw scores)
   ↓
3. Interpretation service generates T-scores, percentiles, categories
   ↓
4. [IF generate_report=true]
   ↓
5. Report service formats data for AI agent
   ↓
6. AI agent (GPT-5 with reasoning) generates narrative report
   ↓
7. All data returned to frontend
   ↓
8. Frontend displays AI report in gradient card above other results
```

### AI Agent Configuration

The AI agent is configured with:

- **Model**: GPT-5
- **Reasoning Effort**: Medium
- **Reasoning Summary**: Auto
- **Store**: True (for traceability)

**Agent Instructions** (Summary):
- Transform structured personality data into natural language
- Use evidence-based psychological terminology
- Avoid numeric values in the narrative
- Write 1-2 paragraphs
- Scientific yet human tone
- Grounded, objective, insightful

### Input Format

The agent receives formatted text like:

```
Generate a personality report based on the following Big Five trait analysis:

=== TRAIT SCORES ===
Extraversion: T-score 62.34 (Percentile 84.5%) - High - Socially engaging, outgoing
Neuroticism: T-score 38.12 (Percentile 23.1%) - Low - Emotionally stable, calm
...

=== PROFILE SUMMARY ===
Mean T-score: 49.87
Dominant traits (T≥60): Extraversion, Conscientiousness
Subdued traits (T≤40): Neuroticism

Context: This analysis is based on photo-based personality inference using a validated ML model.
```

### Output Example

```
The individual's behavioral patterns suggest a highly structured and socially engaging personality profile.
They demonstrate strong adaptive emotional regulation, characterized by low reactivity to stress and
balanced affective states. Their interpersonal dynamics reflect consistent social attunement and goal-oriented
behavior, typical of individuals who thrive in collaborative yet structured environments. In professional
contexts, this profile suggests reliability in team-based settings combined with effective emotional resilience
under demanding conditions.
```

## API Usage

### Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict/upload?generate_report=true" \
  -H "accept: application/json" \
  -F "file=@person.jpg" \
  -F "use_tta=true" \
  -F "method=face+middle" \
  -F "include_interpretations=true"
```

### Response

```json
{
  "success": true,
  "predictions": {
    "extraversion": 0.5234,
    "neuroticism": 0.3456,
    ...
  },
  "interpretations": {
    "extraversion": {
      "raw_score": 0.5234,
      "t_score": 62.34,
      "percentile": 84.5,
      "category": "High",
      "label": "Socially engaging, outgoing",
      "interpretation": "This individual tends to be outgoing..."
    },
    ...
  },
  "summary": {
    "mean_t_score": 49.87,
    "dominant_traits": ["extraversion", "conscientiousness"],
    "subdued_traits": ["neuroticism"]
  },
  "narrative_report": "The individual's behavioral patterns suggest...",
  "metadata": {
    "preprocessing": {...},
    "prediction": {...},
    "norms": {...},
    "cutoffs": {...}
  },
  "timestamp": "2025-01-19T..."
}
```

## Frontend Integration

The frontend automatically:

1. Requests AI report when user clicks "Analyze"
2. Displays loading spinner during generation
3. Shows report in beautiful gradient card
4. Includes report in downloadable text file
5. Handles errors gracefully

### CSS Styling

The AI report card features:
- Purple gradient background (matching main theme)
- White text with semi-transparent backdrop
- Icon header
- Justified text alignment
- Box shadow for depth
- Error state with different gradient (pink/red)

## Error Handling

If report generation fails:

1. `narrative_report` field will be `null`
2. `report_error` field contains error message
3. Frontend shows error state with red gradient
4. Rest of analysis (scores, interpretations) still available
5. Error logged in backend for debugging

## Performance Considerations

- **Caching**: Reports are cached with prediction results
- **Async**: Report generation runs asynchronously
- **Non-blocking**: Prediction and interpretation complete first
- **Timeout**: Agent has built-in timeout protection
- **Optional**: Feature is opt-in via `generate_report` parameter

## Installation

### Dependencies

```bash
pip install anthropic-agents>=0.1.0
pip install openai>=1.0.0
pip install scipy>=1.10.0
```

Or use the provided requirements file:

```bash
pip install -r requirements_ai_agent.txt
```

### Environment Variables

Make sure you have API keys configured (if required by the agents SDK):

```bash
# .env file
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # If required
```

## Testing

### Manual Test

```python
# test_ai_report.py
import asyncio
from src.services.report_service import get_report_service

async def test():
    service = get_report_service()

    interpretations = {
        "extraversion": {
            "t_score": 62.34,
            "percentile": 84.5,
            "category": "High",
            "label": "Socially engaging"
        },
        # ... other traits
    }

    summary = {
        "mean_t_score": 49.87,
        "dominant_traits": ["extraversion"],
        "subdued_traits": ["neuroticism"]
    }

    report = await service.generate_report(interpretations, summary)
    print(report)

asyncio.run(test())
```

### Full Pipeline Test

```bash
# With report
python test_interpretation.py  # Modify to add generate_report=true

# Or via API
curl -X POST "http://localhost:8000/api/v1/predict/upload?generate_report=true&include_interpretations=true" \
  -F "file=@data/image.jpg"
```

## Customization

### Modify Agent Instructions

Edit `src/services/report_service.py` and modify the `instructions` parameter in the `Agent()` constructor.

### Change Model

Update the `model` parameter:

```python
Agent(
    name="Personality Agent",
    instructions="...",
    model="gpt-4",  # or "claude-3-5-sonnet", etc.
    ...
)
```

### Adjust Reasoning Effort

Modify `ModelSettings`:

```python
ModelSettings(
    store=True,
    reasoning=Reasoning(
        effort="low",  # or "high"
        summary="auto"
    )
)
```

## Limitations

1. **API Costs**: Each report generation calls an LLM API
2. **Latency**: Adds 2-5 seconds to response time
3. **Rate Limits**: Subject to API provider rate limits
4. **Accuracy**: Report quality depends on AI model capabilities
5. **Language**: Currently English only

## Best Practices

1. **Enable selectively**: Default `generate_report=false` for performance
2. **Cache aggressively**: Cache includes report in cache key
3. **Monitor costs**: Track API usage and costs
4. **Validate output**: Review generated reports for quality
5. **Provide disclaimers**: Remind users this is AI-generated content

## Future Enhancements

- [ ] Multi-language support
- [ ] Customizable report length
- [ ] Different report styles (clinical, casual, professional)
- [ ] Report templates for specific use cases
- [ ] Batch report generation
- [ ] Report quality scoring
- [ ] User feedback collection

## Support

For issues related to:
- **Agent SDK**: Check Anthropic Agents documentation
- **Report quality**: Adjust agent instructions
- **Performance**: Review caching and async implementation
- **Integration**: Review this documentation and code comments

---

**Generated**: 2025-01-19
**Version**: 1.0
**Author**: OCEAN API Development Team
