# AI Personality Analysis Features

## Overview

The OCEAN API provides comprehensive AI-powered personality analysis features including:
- **Video/Image Analysis**: ML-based personality trait prediction from facial features
- **Voice Analysis**: Audio-based personality insights from speech patterns
- **AI Personality Coach (Firasa)**: Interactive chat with personalized coaching
- **Cultural Fit Analysis**: Workplace culture matching based on personality
- **Job Matching**: Real job search with personality-based scoring
- **CV Upload**: Enhanced job matching from resume data

## Analysis Page Sections

The results page displays personality insights across 10 main sections:

### 1. Big 5 Score (`big5Section`)
- Radar chart visualization of OCEAN scores
- Profile type identification (e.g., "The Creative Leader")
- Individual trait bars with percentile scores
- Shareable personality card

### 2. Unique Story (`uniqueStorySection`)
- AI-generated narrative personality report
- "You are" strengths section
- "Room for improvement" section
- Personalized insights based on trait combinations

### 3. Relationship & Empathy (`relationshipsSection`)
- Empathy Style gauge
- Social Battery gauge
- Trust Building gauge
- Accordion with detailed insights on interpersonal dynamics

### 4. Focus & Execution Style (`workSection`)
- Task Approach gauge
- Detail Orientation gauge
- Decision Style gauge
- Work DNA insights and suitable work environments

### 5. Ideation & Creative Energy (`creativitySection`)
- Idea Generation gauge
- Risk Tolerance gauge
- Adaptability gauge
- Creative process insights

### 6. Pressure Response & Recovery (`stressSection`)
- Stress Tolerance gauge
- Recovery Speed gauge
- Emotional Regulation gauge
- Coping strategies and resilience insights

### 7. Openness to Experience (`opennessSection`)
- Curiosity Level gauge
- Aesthetic Sensitivity gauge
- Intellectual Engagement gauge
- Exploration tendencies

### 8. Learning & Growth (`learningSection`)
- Learning Style gauge
- Feedback Reception gauge
- Growth Mindset gauge
- Personal development insights

### 9. Voice & Communication (`voiceSection` / `audioSection`)
- Only displayed for video recordings with audio
- Speech pace analysis
- Tone confidence gauge
- Communication style insights
- Accordion with voice personality details

### 10. Similarity to Famous (`similaritySection`)
- Top 3 similar famous personalities
- Similarity percentages based on OCEAN profile matching
- Brief descriptions of each personality

## Architecture

### Backend Services

1. **Prediction Service** (`src/services/prediction_service.py`)
   - ML model for personality trait prediction from images/video
   - Audio feature extraction for voice analysis
   - T-score and percentile calculations

2. **AI Agent Service** (`src/services/ai_agent_service.py`)
   - Firasa personality coach using Pydantic AI
   - Multi-provider support: OpenAI, Gemini, Anthropic, Vertex AI
   - Tool-based architecture for cultural fit, job search, CV data

3. **Cultural Fit Service** (`src/services/cultural_fit_service.py`)
   - 8-dimension cultural preference model
   - 12 workplace culture type matching
   - Cosine similarity scoring

4. **CV Parser Service** (`src/services/cv_parser_service.py`)
   - PDF extraction via PyMuPDF
   - DOCX extraction via python-docx
   - LLM-based structured data extraction

5. **LLM Provider** (`src/services/llm_provider.py`)
   - Abstraction layer for multiple AI providers
   - Supports OpenAI, Gemini, Anthropic, Vertex AI
   - Fallback for models without function calling

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/upload` | POST | Analyze uploaded image |
| `/api/v1/predict/video` | POST | Analyze uploaded video |
| `/api/v1/chat/stream` | POST | Stream chat with AI coach |
| `/api/v1/chat/cultural-fit` | POST | Get cultural fit analysis |
| `/api/v1/chat/job-results` | GET | Fetch job search results |
| `/api/v1/chat/upload-cv` | POST | Upload and parse CV |

### Frontend Components

1. **app.js** - Main application logic
   - `displayResults()` - Orchestrates all section rendering
   - `displayBig5Section()` - Radar chart and trait bars
   - `displayUniqueStorySection()` - AI narrative report
   - `displayRelationshipsSection()` - Empathy gauges
   - `displayWorkSection()` - Work style gauges
   - `displayCreativitySection()` - Creative energy gauges
   - `displayStressSection()` - Resilience gauges
   - `displayOpennessSection()` - Openness gauges
   - `displayLearningSection()` - Growth gauges
   - `displayVoiceSection()` - Voice analysis (video only)
   - `displaySimilaritySection()` - Famous personality matches
   - `sendChatMessage()` - AI coach streaming chat

2. **index.html** - Page structure
   - Upload section with drag-drop
   - Video recording with questions
   - Results section with navigation tabs
   - Chat panel with Firasa coach
   - Job results sidebar

3. **style.css** - Styling
   - Gauge components with SVG arcs
   - Accordion animations
   - Chat message bubbles
   - Job cards with personality scores

## AI Coach (Firasa)

### Features
- Personality-based coaching and advice
- Cultural fit analysis on request
- Real job search with matching scores
- CV data integration for enhanced recommendations

### Tools Available to Agent
1. `get_ocean_scores` - Access user's personality scores
2. `get_derived_metrics` - Get computed personality metrics
3. `get_trait_interpretation` - Detailed trait insights
4. `get_cultural_fit` - Workplace culture matching
5. `get_career_profile` - CV data if uploaded
6. `search_matching_jobs` - Real job listings via SerpAPI

### System Prompt Configuration
Located in `src/services/ai_agent_service.py`:
- Job search trigger rules
- Response style guidelines
- Score interpretation scales
- Cultural fit explanation format

## Configuration

### Environment Variables

```bash
# AI Provider (choose one)
AI_PROVIDER=gemini  # or openai, anthropic, vertex
AI_REPORT_ENABLED=true

# OpenAI
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini

# Gemini
GOOGLE_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash-exp

# Anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest

# Vertex AI (Google Cloud)
VERTEX_PROJECT=your_project_id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.0-flash-001

# Job Search
JOB_SEARCH_ENABLED=true
SERPAPI_KEY=your_serpapi_key
```

### Dependencies

```toml
# pyproject.toml
pydantic-ai = "^0.0.39"
google-genai = "^1.0.0"
scipy = "^1.10.0"
python-docx = "^1.1.0"
PyMuPDF = "^1.24.0"
```

## Data Flow

### Personality Analysis Flow
```
1. User uploads image/video or records video
   ↓
2. Backend extracts frames and audio (if video)
   ↓
3. ML model predicts raw OCEAN scores
   ↓
4. Interpretation service calculates T-scores, percentiles
   ↓
5. Derived metrics computed (empathy, creativity, etc.)
   ↓
6. AI generates narrative report
   ↓
7. Frontend displays all sections with animations
```

### Chat Flow
```
1. User types message in chat panel
   ↓
2. Frontend sends POST to /chat/stream with:
   - Message
   - OCEAN scores
   - Derived metrics
   - Message history
   - Career profile (if CV uploaded)
   ↓
3. Agent processes with available tools
   ↓
4. Response streamed via SSE
   ↓
5. If job search triggered:
   - Jobs stored server-side
   - Frontend fetches via /job-results
   - Job cards displayed in sidebar
```

### CV Upload Flow
```
1. User clicks "Upload CV" in quick actions
   ↓
2. File uploaded to /chat/upload-cv
   ↓
3. Text extracted (PDF via PyMuPDF, DOCX via python-docx)
   ↓
4. LLM parses to structured CareerProfile
   ↓
5. Profile stored in frontend state
   ↓
6. Subsequent job searches use CV data automatically
```

## Error Handling

- **AI Service Unavailable**: Falls back to static interpretations
- **Job Search Disabled**: Informs user, no search attempted
- **CV Parse Failure**: Returns error message, chat continues
- **Streaming Error**: Error event sent, chat recovers

## Performance Considerations

- **Caching**: Prediction results cached by file hash
- **Streaming**: Chat responses stream in real-time via SSE
- **Lazy Loading**: Voice section only renders for video uploads
- **Image Optimization**: Frames extracted at optimal intervals

## Testing

### Manual Testing
```bash
# Start server
uvicorn src.api.main:app --reload

# Test image analysis
curl -X POST "http://localhost:8000/api/v1/predict/upload" \
  -F "file=@test_image.jpg" \
  -F "include_interpretations=true"

# Test chat
curl -X POST "http://localhost:8000/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are my strengths?", "ocean_scores": {...}}'
```

### Frontend Testing
1. Upload image → All sections should display
2. Record video → Voice section should appear
3. Chat with Firasa → Streaming responses
4. Request job matching → Job cards in sidebar
5. Upload CV → Enhanced job recommendations

---

**Last Updated**: 2025-02
**Version**: 2.0
