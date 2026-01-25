# OCEAN Personality API Contract

## Base URL
```
http://localhost:8001
```

## Interactive Documentation
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI JSON**: http://localhost:8001/openapi.json

---

## Main Endpoint

### `POST /api/v1/predict/upload`

Upload a video for personality analysis with optional gamified assessment metadata.

#### Request

**Content-Type**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Video file (.mp4, .avi, .mov, .mkv, .webm) |
| `method` | Query | No | Face detection: `face`, `middle`, `face+middle` (default) |
| `use_tta` | Query | No | Test Time Augmentation (default: true) |
| `include_interpretations` | Query | No | Include T-score interpretations (default: true) |
| `generate_report` | Query | No | Generate AI insights (default: false) |
| `question_responses` | Form | No | JSON string with assessment metadata |

#### Assessment Metadata Schema (question_responses)

When using the gamified Superpower Question assessment:

```json
{
  "compound_question": {
    "full_question": "The Superpower Question",
    "parts": [
      {
        "id": "superpower_what",
        "prompt": "If you had a superpower, what would it be?",
        "trait": "Openness",
        "signals": ["creativity", "imagination", "novelty-seeking"]
      },
      {
        "id": "superpower_use",
        "prompt": "How would you use this superpower in your daily life?",
        "trait": "Conscientiousness",
        "signals": ["planning", "structure", "follow-through"]
      },
      {
        "id": "superpower_fails",
        "prompt": "What would you do when your superpower doesn't work?",
        "trait": "Neuroticism",
        "signals": ["stress response", "resilience", "coping"]
      }
    ]
  },
  "question_responses": [
    {
      "question_id": "superpower_what",
      "question_text": "If you had a superpower, what would it be?",
      "skipped": false,
      "start_time": 0.0,
      "end_time": 45.0,
      "duration": 45.0
    },
    {
      "question_id": "superpower_use",
      "question_text": "How would you use this superpower in your daily life?",
      "skipped": false,
      "start_time": 45.0,
      "end_time": 90.0,
      "duration": 45.0
    },
    {
      "question_id": "superpower_fails",
      "question_text": "What would you do when your superpower doesn't work?",
      "skipped": false,
      "start_time": 90.0,
      "end_time": 135.0,
      "duration": 45.0
    }
  ],
  "total_questions": 3,
  "questions_answered": 3,
  "questions_skipped": 0
}
```

---

## Response Schema

### `PredictionResponse`

```typescript
{
  success: boolean;
  predictions: PersonalityTraits;
  interpretations?: object;           // T-score interpretations
  summary?: object;                   // Profile summary
  insights?: PersonalityInsights;     // AI-generated overview
  relationship_metrics?: RelationshipMetrics;
  work_metrics?: WorkMetrics;
  creativity_metrics?: CreativityMetrics;
  stress_metrics?: StressMetrics;
  audio_metrics?: AudioMetrics;
  transcript?: string;                // Video transcription
  report_error?: string;              // Error if report generation failed
  metadata?: object;
  timestamp: string;                  // ISO 8601
}
```

---

### Nested Types

#### `PersonalityTraits`
OCEAN scores ranging from -1.0 to 1.0

```typescript
{
  openness: number;          // -1.0 to 1.0
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}
```

---

#### `PersonalityInsights`
AI-generated personality overview

```typescript
{
  title: string;              // e.g., "The Creative Strategist"
  tags: TraitTag[];           // 3 trait tags
  description: string;        // 1-2 sentence summary
  quote: string;              // Personal motto (10-15 words)
  story: string;              // Narrative (100-150 words)
  story_traits: TraitTag[];   // 6-8 "You are..." descriptors
}

// TraitTag
{
  emoji: string;              // e.g., "🎯"
  label: string;              // e.g., "Purpose-driven"
}
```

---

#### `RelationshipMetrics`
Relationship and empathy insights

```typescript
{
  metrics: {
    trust_signaling: DerivedMetric;
    social_openness: DerivedMetric;
    empathic_disposition: DerivedMetric;
    conflict_avoidance: DerivedMetric;
    harmony_seeking: DerivedMetric;
    anxiety_avoidance: DerivedMetric;
  };
  coach_recommendation: string;       // 50-100 words
  actionable_steps: ActionableStep[]; // 6 steps

  // Accordion fields
  snapshot_insight?: string;
  behavioral_patterns?: BehavioralPattern[];
  how_others_experience?: string;
  strength?: StrengthTradeoff;
  tradeoff?: StrengthTradeoff;
  growth_lever?: string;
  suitable_for?: string[];
}
```

---

#### `WorkMetrics`
Work and productivity insights

```typescript
{
  metrics: {
    persistence: DerivedMetric;
    focus_attention: DerivedMetric;
    structure_preference: DerivedMetric;
    risk_aversion: DerivedMetric;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];

  // Accordion fields (same as RelationshipMetrics)
  snapshot_insight?: string;
  behavioral_patterns?: BehavioralPattern[];
  how_others_experience?: string;
  strength?: StrengthTradeoff;
  tradeoff?: StrengthTradeoff;
  growth_lever?: string;
  suitable_for?: string[];
}
```

---

#### `CreativityMetrics`
Creativity and innovation insights

```typescript
{
  metrics: {
    ideation_power: DerivedMetric;
    openness_to_novelty: DerivedMetric;
    originality_index: DerivedMetric;
    attention_to_detail_creative: DerivedMetric;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];

  // Accordion fields (same structure)
  ...
}
```

---

#### `StressMetrics`
Stress and resilience insights

```typescript
{
  metrics: {
    stress_indicators: DerivedMetric;
    emotional_regulation: DerivedMetric;
    resilience_score: DerivedMetric;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];

  // Accordion fields (same structure)
  ...
}
```

---

#### `AudioMetrics`
Voice and speech analysis

```typescript
{
  indicators: {
    vocal_extraversion: VocalIndicator;
    vocal_stability: VocalIndicator;
    vocal_confidence: VocalIndicator;
    vocal_warmth: VocalIndicator;
  };
  interpretations: {
    pitch: string;
    expressiveness: string;
    volume: string;
    pace: string;
    brightness: string;
    stability: string;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];
}

// VocalIndicator
{
  score: number;        // 0-100
  level: string;        // "Low" | "Moderate" | "High"
  signals: string[];    // Observable vocal signals
}
```

---

#### Shared Types

```typescript
// DerivedMetric
{
  score: number;        // 0-100
  level: string;        // "Low" | "Moderate" | "High"
  description: string;  // Human-readable explanation
}

// ActionableStep
{
  emoji: string;        // e.g., "🎯"
  text: string;         // Action description
}

// BehavioralPattern
{
  title: string;
  description: string;
}

// StrengthTradeoff
{
  title: string;
  description: string;
}
```

---

## Example Response

```json
{
  "success": true,
  "predictions": {
    "openness": 0.65,
    "conscientiousness": 0.78,
    "extraversion": 0.45,
    "agreeableness": 0.52,
    "neuroticism": -0.23
  },
  "insights": {
    "title": "The Creative Strategist",
    "tags": [
      {"emoji": "🎨", "label": "Imaginative"},
      {"emoji": "📋", "label": "Organized"},
      {"emoji": "🌿", "label": "Calm"}
    ],
    "description": "You chose time manipulation as your superpower, revealing a creative mind that values control and planning. Your structured approach to using it daily shows conscientiousness.",
    "quote": "Time bends to those who plan with purpose and dream with imagination.",
    "story": "When asked about a superpower, you didn't hesitate—time manipulation. But it wasn't the power itself that revealed your personality; it was how you planned to use it. You described waking up earlier to get more done, revisiting conversations to say the right thing, and giving yourself space to think. This isn't escapism—it's optimization. Your response to failure was equally telling: you'd simply try again, learn from the mistake, and adapt. No catastrophizing, no drama—just practical problem-solving.",
    "story_traits": [
      {"emoji": "🎯", "label": "Purpose-driven"},
      {"emoji": "🧠", "label": "Strategic thinker"},
      {"emoji": "🔄", "label": "Adaptable"},
      {"emoji": "📊", "label": "Detail-oriented"},
      {"emoji": "🌊", "label": "Emotionally steady"},
      {"emoji": "💡", "label": "Innovative"}
    ]
  },
  "relationship_metrics": {
    "metrics": {
      "trust_signaling": {
        "score": 75,
        "level": "High",
        "description": "Your calm demeanor and thoughtful responses inspire confidence in others."
      },
      "social_openness": {
        "score": 62,
        "level": "Moderate",
        "description": "You engage comfortably in social settings while maintaining healthy boundaries."
      }
    },
    "coach_recommendation": "Your thoughtful communication style—pausing before answering, speaking clearly—creates a sense of reliability. You mentioned wanting to 'revisit conversations,' suggesting you care about how you come across. This self-awareness is valuable in relationships. Consider sharing your thoughts more spontaneously sometimes; not everything needs to be optimized.",
    "actionable_steps": [
      {"emoji": "💬", "text": "Practice spontaneous conversations without pre-planning"},
      {"emoji": "👂", "text": "Ask follow-up questions to show genuine interest"},
      {"emoji": "🤝", "text": "Share vulnerabilities to deepen connections"}
    ],
    "snapshot_insight": "You approach relationships with the same thoughtfulness you bring to everything else—carefully, deliberately, and with intention.",
    "strength": {
      "title": "Reliable Presence",
      "description": "People know what to expect from you, which builds lasting trust."
    },
    "tradeoff": {
      "title": "Emotional Distance",
      "description": "Your composed nature may sometimes read as detachment."
    }
  },
  "work_metrics": {
    "metrics": {
      "persistence": {"score": 82, "level": "High", "description": "You show strong follow-through."},
      "structure_preference": {"score": 78, "level": "High", "description": "You thrive with clear processes."}
    },
    "coach_recommendation": "Your answer was notably structured—you described a clear sequence of how you'd use time manipulation. This reveals a mind that naturally organizes information. When your 'superpower fails,' you said you'd 'figure out another way,' showing problem-solving orientation over frustration."
  },
  "creativity_metrics": {
    "metrics": {
      "ideation_power": {"score": 70, "level": "Moderate", "description": "Solid creative capacity with practical grounding."}
    },
    "coach_recommendation": "Time manipulation is a conceptually sophisticated superpower choice—it requires abstract thinking about causality and consequence. You didn't just say 'I'd stop time'; you described specific use cases. This suggests creativity channeled through practicality."
  },
  "stress_metrics": {
    "metrics": {
      "stress_indicators": {"score": 28, "level": "Low", "description": "You handle pressure with composure."},
      "resilience_score": {"score": 76, "level": "High", "description": "Strong bounce-back capacity."}
    },
    "coach_recommendation": "When asked what you'd do if your superpower failed, you showed no signs of distress—no furrowed brow, no hesitation. Your answer was pragmatic: 'I'd try something else.' This calm problem-solving under hypothetical adversity suggests genuine resilience, not just stated resilience."
  },
  "audio_metrics": {
    "indicators": {
      "vocal_confidence": {"score": 68, "level": "Moderate", "signals": ["steady pace", "clear articulation"]}
    },
    "interpretations": {
      "pace": "Measured speaking pace with thoughtful pauses",
      "expressiveness": "Moderate vocal range, controlled delivery"
    }
  },
  "transcript": "If I had a superpower, I think I'd choose time manipulation. Not stopping time exactly, but more like being able to slow it down or rewind small moments. I'd use it in my daily life to give myself more time to think before responding to things, or to redo conversations that didn't go well. If it stopped working, I'd probably just... figure out another way. I mean, I've lived without it this long, right? I'd adapt.",
  "metadata": {
    "multimodal": {
      "frames_analyzed": 10,
      "has_transcript": true,
      "transcript_length": 412,
      "has_assessment": true,
      "questions_answered": 3
    }
  },
  "timestamp": "2026-01-13T12:00:00Z"
}
```

---

## Other Endpoints

### `GET /api/v1/health`
Health check

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "timestamp": "2026-01-13T12:00:00Z"
}
```

### `GET /api/v1/traits`
Get trait descriptions

```json
{
  "traits": {
    "openness": "Creative, curious, open to new experiences",
    "conscientiousness": "Organized, disciplined, responsible",
    "extraversion": "Outgoing, social, energetic",
    "agreeableness": "Cooperative, trusting, helpful",
    "neuroticism": "Anxious, moody, emotionally unstable"
  }
}
```

### `WebSocket /ws`
Real-time OCEAN score streaming during video recording

---

## Streaming Endpoint (SSE)

### `POST /api/v1/stream/analyze`

**Real-time progress updates via Server-Sent Events (SSE)**

Same parameters as `/api/v1/predict/upload`, but returns a stream of progress events instead of waiting for the full result.

#### Response Format

**Content-Type**: `text/event-stream`

Each event is a JSON object with:

```typescript
{
  stage: string;      // Current processing stage
  progress: number;   // 0-100 percentage
  message: string;    // Human-readable status
  timestamp: number;  // Unix timestamp
  result?: object;    // Full result (only on "complete" stage)
}
```

#### Progress Stages

| Stage | Progress | Description |
|-------|----------|-------------|
| `uploading` | 0-5% | Receiving and validating video |
| `compressing` | 5-15% | Compressing video if needed |
| `extracting_frames` | 15-30% | Extracting frames for analysis |
| `transcribing` | 30-50% | Transcribing audio to text |
| `analyzing_video` | 50-65% | Running OCEAN prediction |
| `analyzing_audio` | 65-75% | Analyzing voice patterns |
| `generating_report` | 75-95% | Generating AI insights |
| `complete` | 100% | Analysis complete with result |
| `error` | -1 | Error occurred |

#### Example SSE Stream

```
data: {"stage":"uploading","progress":0,"message":"Receiving video...","timestamp":1705312800}

data: {"stage":"uploading","progress":5,"message":"Video received (2.5MB)","timestamp":1705312801}

data: {"stage":"compressing","progress":15,"message":"No compression needed","timestamp":1705312802}

data: {"stage":"extracting_frames","progress":20,"message":"Extracting video frames...","timestamp":1705312803}

data: {"stage":"transcribing","progress":35,"message":"Transcribing audio...","timestamp":1705312805}

data: {"stage":"analyzing_video","progress":55,"message":"Analyzing video frames...","timestamp":1705312810}

data: {"stage":"analyzing_video","progress":65,"message":"Video analysis complete","timestamp":1705312815}

data: {"stage":"analyzing_audio","progress":70,"message":"Analyzing voice patterns...","timestamp":1705312816}

data: {"stage":"generating_report","progress":85,"message":"Running AI analysis...","timestamp":1705312820}

data: {"stage":"complete","progress":100,"message":"Analysis complete","timestamp":1705312825,"result":{...full response...}}
```

#### JavaScript Client Example

```javascript
async function analyzeWithProgress(videoFile, onProgress) {
    const formData = new FormData();
    formData.append('file', videoFile);

    const response = await fetch('/api/v1/stream/analyze?generate_report=true', {
        method: 'POST',
        body: formData
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));

                // Update progress UI
                onProgress(data.progress, data.message);

                if (data.stage === 'complete') {
                    return data.result;  // Full analysis result
                }

                if (data.stage === 'error') {
                    throw new Error(data.message);
                }
            }
        }
    }
}

// Usage
analyzeWithProgress(videoFile, (progress, message) => {
    progressBar.style.width = `${progress}%`;
    statusText.textContent = message;
}).then(result => {
    console.log('Analysis complete:', result);
}).catch(err => {
    console.error('Analysis failed:', err);
});
```

### `GET /api/v1/stream/test`

Test SSE endpoint - sends simulated progress updates every 0.5 seconds.
Useful for testing your SSE client implementation.

---

## Error Response

```json
{
  "success": false,
  "error": "No face detected in video",
  "error_type": "FaceDetectionError",
  "details": {},
  "timestamp": "2026-01-13T12:00:00Z"
}
```

---

## Notes

1. **OCEAN scores** range from -1.0 to 1.0 (normalized from model output)
2. **Derived metrics** are 0-100 percentage scores with Low/Moderate/High levels
3. **Coaching content** is generated from transcript + video frames + voice (NOT from OCEAN scores)
4. **Assessment metadata** enables the LLM to analyze responses in context of the questions asked
5. **Audio metrics** require video with audio; silent videos will have null audio_metrics
