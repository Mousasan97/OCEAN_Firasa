# OCEAN Personality Analysis API Documentation

## Base URL

```
https://your-domain.com/api/v1
```

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/upload` | POST | Analyze uploaded image |
| `/predict/video` | POST | Analyze uploaded video |
| `/chat/stream` | POST | Stream chat with AI coach |
| `/chat/cultural-fit` | POST | Get cultural fit analysis |
| `/chat/job-results` | GET | Fetch job search results |
| `/chat/upload-cv` | POST | Upload and parse CV |
| `/chat/health` | GET | Check chat service status |
| `/health` | GET | API health check |

---

## Frontend Sections Overview

The analysis results page displays data across these sections:

| Section | Navigation Label | Data Source |
|---------|------------------|-------------|
| Big 5 Score | `big5Section` | `insights`, `interpretations`, `predictions` |
| Unique Story | `uniqueStorySection` | `insights.quote`, `insights.story`, `insights.story_traits` |
| Relationship & Empathy | `relationshipsSection` | `relationship_metrics` |
| Focus & Execution Style | `workSection` | `work_metrics` |
| Ideation & Creative Energy | `creativitySection` | `creativity_metrics` |
| Pressure Response & Recovery | `stressSection` | `stress_metrics` |
| Openness to Experience | `opennessSection` | `openness_metrics` |
| Learning & Growth | `learningSection` | `learning_metrics` |
| Voice & Communication | `voiceSection` | `audio_metrics` (video only) |
| Similarity to Famous | `similaritySection` | *Currently disabled* |

---

## Prediction Endpoints

### POST `/predict/upload`

Upload an image for personality analysis.

**Request:**
```
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Image file (JPG, PNG, WebP) |
| `include_interpretations` | boolean | No | Include T-score interpretations (default: true) |
| `generate_report` | boolean | No | Generate AI insights (default: true) |

**Example:**
```bash
curl -X POST "https://api.example.com/api/v1/predict/upload" \
  -F "file=@photo.jpg" \
  -F "include_interpretations=true" \
  -F "generate_report=true"
```

---

### POST `/predict/video`

Upload a video for comprehensive personality analysis including voice analysis.

**Request:**
```
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Video file (MP4, AVI, MOV, WebM) |
| `include_interpretations` | boolean | No | Include T-score interpretations (default: true) |
| `generate_report` | boolean | No | Generate AI insights (default: true) |
| `assessment_metadata` | JSON string | No | Question timestamps for gamified assessment |

**Example:**
```bash
curl -X POST "https://api.example.com/api/v1/predict/video" \
  -F "file=@recording.mp4" \
  -F "include_interpretations=true" \
  -F "generate_report=true"
```

**With Assessment Metadata (Gamified Flow):**
```bash
curl -X POST "https://api.example.com/api/v1/predict/video" \
  -F "file=@recording.mp4" \
  -F 'assessment_metadata={
    "question_responses": [
      {
        "question_id": "openness",
        "question_text": "If you had a superpower...",
        "skipped": false,
        "start_time": 0.0,
        "end_time": 38.5,
        "duration": 38.5
      }
    ],
    "total_questions": 5,
    "questions_answered": 4,
    "questions_skipped": 1
  }'
```

---

### `AssessmentMetadata` (Gamified Recording)

Used when video is recorded via the question-based assessment flow.

```typescript
{
  question_responses: QuestionResponse[];
  total_questions: number;        // Always 5
  questions_answered: number;
  questions_skipped: number;
}

// QuestionResponse
{
  question_id: string;            // "openness" | "conscientiousness" | "extraversion" | "agreeableness" | "neuroticism"
  question_text: string;          // The question shown to user
  skipped: boolean;
  start_time: number;             // Seconds into video
  end_time: number;
  duration: number;               // Response duration in seconds
}
```

---

---

## Section Details

### Big 5 Score Section
Displays:
- **Insights Card**: Profile photo/video, `insights.title`, `insights.description`
- **Circular Score Badges**: 5 OCEAN scores from `interpretations[trait].t_score`
- **Radar Chart**: Visualization of all 5 T-scores
- **Trait Accordion**: Expandable details for each trait with interpretation text
- **Shareable Cards**: Social media cards with scores and photo

### Unique Story Section
Displays:
- **Quote**: `insights.quote` (10-15 word personal motto)
- **Narrative**: `insights.story` (100-150 word personality narrative)
- **"You are" Tags**: `insights.story_traits` (6-8 trait descriptors with emojis)
- **Room for Improvement**: Static improvement tips

### Metric Sections (Relationships, Work, Creativity, Stress, Openness, Learning)
Each displays:
- **3-4 Gauge Charts**: From `metrics` object (score 0-100, level, description)
- **Coach Recommendation**: `coach_recommendation` text
- **Accordion Items**:
  - Snapshot: `snapshot_insight`
  - Behavioral Patterns: `behavioral_patterns[]`
  - How Others Experience You: `how_others_experience`
  - Strength: `strength.title` + `strength.description`
  - Tradeoff: `tradeoff.title` + `tradeoff.description`
  - Growth Lever: `growth_lever`
- **Suitable For Tags**: `suitable_for[]` array

### Voice & Communication Section (Video Only)
Only displayed when `audio_metrics` is present:
- **Vocal Indicator Gauges**: 4 indicators (extraversion, stability, confidence, warmth)
- **Interpretations**: pitch, expressiveness, volume, pace, brightness, stability
- **Coach Recommendation**: Voice coaching tips
- **Actionable Steps**: `actionable_steps[]`

### Debug Panel (Development Only)
When `metadata.debug_visualization` is present:
- **Extracted Frames**: Base64 images with face detection boxes
- **Transcript**: Full text and character count
- **Audio Waveform**: Canvas visualization with duration

---

## Response Schema

### `PredictionResponse`

```typescript
{
  success: boolean;
  predictions: PersonalityTraits;
  interpretations?: TraitInterpretations;
  summary?: ProfileSummary;
  insights?: PersonalityInsights;
  relationship_metrics?: RelationshipMetrics;
  work_metrics?: WorkMetrics;
  creativity_metrics?: CreativityMetrics;
  stress_metrics?: StressMetrics;
  openness_metrics?: OpennessMetrics;
  learning_metrics?: LearningMetrics;
  audio_metrics?: AudioMetrics;           // Video only
  transcript?: string;                     // Video only
  report_error?: string;
  metadata?: PredictionMetadata;
  timestamp: string;                       // ISO 8601
}
```

---

## Nested Types

### `PersonalityTraits`

OCEAN scores ranging from 0.0 to 1.0 (internally normalized from -1.0 to 1.0).

```typescript
{
  openness: number;          // 0.0 to 1.0
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}
```

---

### `TraitInterpretations`

T-score interpretations for each trait.

```typescript
{
  openness: TraitInterpretation;
  conscientiousness: TraitInterpretation;
  extraversion: TraitInterpretation;
  agreeableness: TraitInterpretation;
  neuroticism: TraitInterpretation;
}

// TraitInterpretation
{
  raw_score: number;      // 0.0 to 1.0
  t_score: number;        // ~20-80, mean=50, std=10
  percentile: number;     // 0-100
  category: string;       // "Very Low" | "Low" | "Average" | "High" | "Very High"
  label: string;          // e.g., "Highly curious and imaginative"
  interpretation: string; // Detailed narrative (100+ words)
}
```

**Category Thresholds:**
| Category | T-Score Range |
|----------|---------------|
| Very Low | T ≤ 30 |
| Low | 31 ≤ T ≤ 44 |
| Average | 45 ≤ T ≤ 55 |
| High | 56 ≤ T ≤ 64 |
| Very High | T ≥ 65 |

---

### `ProfileSummary`

Overall personality profile summary.

```typescript
{
  category_distribution: {
    "Very Low": number;
    "Low": number;
    "Average": number;
    "High": number;
    "Very High": number;
  };
  dominant_traits: string[];    // Traits with T ≥ 60
  subdued_traits: string[];     // Traits with T ≤ 40
  mean_t_score: number;         // Average T-score
  total_traits: number;         // Always 5
}
```

---

### `PersonalityInsights`

AI-generated personality overview for the "Big 5 Score" and "Unique Story" sections.

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

### `RelationshipMetrics`

Relationship and empathy insights for the "Relationship & Empathy" section.

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
  suitable_for?: string[];            // e.g., ["Team Environments", "Client-Facing Roles"]
}
```

---

### `WorkMetrics`

Work and productivity insights for the "Focus & Execution Style" section.

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

### `CreativityMetrics`

Creativity and innovation insights for the "Ideation & Creative Energy" section.

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

### `StressMetrics`

Stress and resilience insights for the "Pressure Response & Recovery" section.

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

### `OpennessMetrics`

Openness to experience insights for the "Openness to Experience" section.

```typescript
{
  metrics: {
    openness_to_experience: DerivedMetric;
    novelty_seeking: DerivedMetric;
    risk_tolerance_adventure: DerivedMetric;
    planning_preference: DerivedMetric;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];

  // Accordion fields (same structure)
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

### `LearningMetrics`

Learning and growth insights for the "Learning & Growth" section.

```typescript
{
  metrics: {
    intellectual_curiosity: DerivedMetric;
    reflective_tendency: DerivedMetric;
    structured_learning_preference: DerivedMetric;
    adaptability_index: DerivedMetric;
  };
  coach_recommendation: string;
  actionable_steps: ActionableStep[];

  // Accordion fields (same structure)
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

### `AudioMetrics`

Voice and speech analysis for the "Voice & Communication" section (video only).

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
  signals: string[];    // e.g., ["projects voice confidently", "speaks rapidly"]
}
```

---

### Shared Types

#### `DerivedMetric`

```typescript
{
  score: number;        // 0-100 percentage
  level: string;        // "Low" | "Moderate" | "High"
  description: string;  // Personalized description based on score
}
```

**Level Thresholds:**
| Level | Score Range |
|-------|-------------|
| Low | 0-35 |
| Moderate | 36-64 |
| High | 65-100 |

---

#### `ActionableStep`

```typescript
{
  emoji: string;        // e.g., "🎯"
  text: string;         // e.g., "Set small daily goals"
}
```

---

#### `BehavioralPattern`

```typescript
{
  title: string;        // e.g., "Active Listener"
  description: string;  // e.g., "You naturally pick up on emotional cues..."
}
```

---

#### `StrengthTradeoff`

```typescript
{
  title: string;        // e.g., "Empathy Champion"
  description: string;  // e.g., "Your ability to understand others..."
}
```

---

## Chat Endpoints

### Chat Panel Features

The chat panel includes quick actions for common queries:

| Quick Action | Prompt Sent |
|--------------|-------------|
| My strengths | "What are my strongest personality traits?" |
| Improve weakest trait | "How can I improve my weakest trait?" |
| Cultural fit | "What workplace cultures and environments best match my personality?" |
| Job matching | "Find jobs that match my personality." |
| Daily habit plan | "Give me a daily habit plan based on my personality" |
| Stress management | "What are my stress triggers and how can I manage them?" |
| Upload CV | Opens file dialog for PDF/DOCX upload |

---

### POST `/chat/stream`

Stream a conversation with the AI personality coach (Firasa).

**Request:**
```typescript
{
  message: string;                    // User's message (1-2000 chars)
  ocean_scores: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  derived_metrics?: object;           // Optional derived metrics
  interpretations?: object;           // Optional trait interpretations
  user_transcript?: string;           // Optional video transcript
  message_history?: ChatMessage[];    // Previous messages
  career_profile?: CareerProfile;     // CV data if uploaded
}

// ChatMessage
{
  role: "user" | "assistant";
  content: string;
}
```

**Response:** Server-Sent Events (SSE)

```
data: {"type": "chunk", "content": "Hello! "}
data: {"type": "chunk", "content": "Based on your personality..."}
data: {"type": "done", "jobs_available": false}
```

**Event Types:**
| Type | Description |
|------|-------------|
| `chunk` | Text chunk from AI response |
| `done` | Stream complete, includes `jobs_available` flag |
| `error` | Error occurred during streaming |

---

### POST `/chat/cultural-fit`

Get cultural fit analysis matching personality to workplace cultures.

**Request:**
```typescript
{
  ocean_scores: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
  derived_metrics?: object;
}
```

**Response:**
```typescript
{
  culture_dimensions: {
    innovation: number;      // 0-100
    stability: number;
    collaboration: number;
    competition: number;
    autonomy: number;
    structure: number;
    flexibility: number;
    results_focus: number;
  };
  top_matches: CultureMatch[];  // Top 3-5 culture matches
  recommendations: string[];
}

// CultureMatch
{
  culture_type: string;     // e.g., "Tech Innovator", "Startup Disruptor"
  fit_score: number;        // 0-100
  strengths: string[];
  challenges: string[];
  description: string;
}
```

**Culture Types:**
- Startup Disruptor
- Tech Innovator
- Corporate Enterprise
- Creative Agency
- Mission-Driven
- Consulting
- Remote/Distributed
- Family Business
- Research/Academic
- Healthcare
- Government
- Entrepreneurial

---

### GET `/chat/job-results`

Fetch job search results after AI coach triggers a search.

**Response:**
```typescript
{
  jobs: JobResult[];
  query: {
    role: string;
    location: string;
  };
  total_found: number;
}

// JobResult
{
  title: string;              // Job title
  company: string;            // Company name
  company_logo?: string;      // Company logo URL
  location: string;           // Job location
  salary?: string;            // Salary range if available
  posted?: string;            // Posted date (e.g., "2 days ago")
  apply_link?: string;        // URL to apply
  fit_score: number;          // 0-100 personality match score
  culture_type?: string;      // e.g., "Tech Innovator", "Startup Disruptor"
  why_fits?: string;          // Explanation of why this job matches
}
```

**Job Card Display:**

Jobs are displayed in a sidebar panel with:
- **Fit Score Badge**: Color-coded (green ≥75%, yellow ≥50%, red <50%)
- **Company Logo**: If available
- **Location/Salary/Posted Tags**: Meta information
- **Culture Type**: Matched workplace culture
- **Why It Fits**: Personality-based explanation
- **Apply Button**: Links to job posting

---

### POST `/chat/upload-cv`

Upload and parse a CV for enhanced job matching.

**Request:**
```
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | CV file (PDF, DOCX, DOC) |

**Max file size:** 10 MB

**Response:**
```typescript
{
  success: boolean;
  career_profile?: CareerProfile;
  error?: string;
}

// CareerProfile
{
  current_role?: string;        // e.g., "Senior Software Engineer"
  target_role?: string;         // Desired role if mentioned
  location?: string;            // e.g., "London, UK"
  years_experience?: number;    // Total years
  key_skills: string[];         // Max 10 skills
  industries: string[];         // e.g., ["Technology", "Finance"]
  education_level?: string;     // e.g., "Master's"
  certifications: string[];
  summary?: string;             // 2-3 sentence summary
}
```

---

## Error Responses

### `ErrorResponse`

```typescript
{
  success: false;
  error: string;              // Human-readable error message
  error_type?: string;        // e.g., "FaceDetectionError"
  details?: object;
  timestamp: string;          // ISO 8601
}
```

**Common Error Types:**
| Error Type | Description |
|------------|-------------|
| `FaceDetectionError` | No face detected in image |
| `FileSizeExceededError` | File exceeds size limit |
| `UnsupportedFileTypeError` | File type not supported |
| `ValidationError` | Request validation failed |
| `ServiceUnavailableError` | AI service not configured |

---

## Metadata

### `PredictionMetadata`

Included in prediction responses for debugging and reproducibility.

```typescript
{
  preprocessing: {
    total_frames?: number;
    fps?: number;
    duration_seconds?: number;
    num_frames_extracted?: number;
    extraction_method?: string;     // e.g., "k_segment"
    face_detected?: boolean;
    width?: number;
    height?: number;
  };
  prediction: {
    model: string;                  // "vat" | "resnet"
    num_frames: number;
    tta_used: boolean;
    device: string;                 // "cuda" | "cpu"
  };
  multimodal?: {
    frames_analyzed: number;
    has_transcript: boolean;
    transcript_length: number;
  };
  norms: {
    source: string;
    version: string;
    means: object;
    stds: object;
  };
  cutoffs: {
    very_low: number;
    low_hi: number;
    avg_hi: number;
    high_hi: number;
    very_high: number;
  };
  debug_visualization?: DebugVisualization;  // Development only
}

// DebugVisualization (shown in debug panel)
{
  frames: DebugFrame[];
  transcript?: string;
  transcript_length?: number;
  waveform?: {
    samples: number[];
    duration_seconds: number;
  };
}

// DebugFrame
{
  image_base64: string;           // Base64 encoded image
  face_detected: boolean;
  face_bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  width: number;
  height: number;
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/predict/*` | 10 requests/minute |
| `/chat/*` | 30 requests/minute |

---

## Example Full Response

```json
{
  "success": true,
  "predictions": {
    "openness": 0.72,
    "conscientiousness": 0.58,
    "extraversion": 0.65,
    "agreeableness": 0.71,
    "neuroticism": 0.32
  },
  "interpretations": {
    "openness": {
      "raw_score": 0.72,
      "t_score": 64.67,
      "percentile": 92.3,
      "category": "High",
      "label": "Highly curious and imaginative",
      "interpretation": "You have a strong drive to explore new ideas..."
    }
  },
  "summary": {
    "category_distribution": {
      "Very Low": 0,
      "Low": 1,
      "Average": 1,
      "High": 3,
      "Very High": 0
    },
    "dominant_traits": ["openness", "extraversion", "agreeableness"],
    "subdued_traits": ["neuroticism"],
    "mean_t_score": 56.4,
    "total_traits": 5
  },
  "insights": {
    "title": "The Collaborative Visionary",
    "tags": [
      {"emoji": "🎨", "label": "Creative"},
      {"emoji": "🤝", "label": "Empathetic"},
      {"emoji": "⚡", "label": "Energetic"}
    ],
    "description": "Your personality suggests a naturally curious and socially engaged individual who thrives in collaborative environments.",
    "quote": "Where imagination meets connection, possibilities become reality.",
    "story": "You move through life with an infectious curiosity...",
    "story_traits": [
      {"emoji": "🌟", "label": "Naturally curious"},
      {"emoji": "💬", "label": "Great communicator"},
      {"emoji": "🎯", "label": "Purpose-driven"}
    ]
  },
  "relationship_metrics": {
    "metrics": {
      "trust_signaling": {
        "score": 78,
        "level": "High",
        "description": "Your warmth and reliability naturally inspire trust."
      },
      "social_openness": {
        "score": 72,
        "level": "High",
        "description": "You thrive in social settings and actively seek connections."
      },
      "empathic_disposition": {
        "score": 68,
        "level": "Moderate",
        "description": "You balance empathy with objectivity."
      },
      "conflict_avoidance": {
        "score": 45,
        "level": "Moderate",
        "description": "You can handle confrontation when necessary."
      },
      "harmony_seeking": {
        "score": 62,
        "level": "Moderate",
        "description": "You value harmony while being direct."
      },
      "anxiety_avoidance": {
        "score": 28,
        "level": "Low",
        "description": "You handle stressful situations with composure."
      }
    },
    "coach_recommendation": "Your natural warmth and reliability are powerful assets...",
    "actionable_steps": [
      {"emoji": "🤝", "text": "Join communities aligned with your interests"},
      {"emoji": "👂", "text": "Practice reflective listening"},
      {"emoji": "💬", "text": "Share your perspective more openly"}
    ],
    "snapshot_insight": "You show up as warm, engaged, and trustworthy...",
    "suitable_for": ["Team Leadership", "Client Relations", "Mentoring"]
  },
  "timestamp": "2025-02-04T12:30:00Z"
}
```

---

**Last Updated:** 2025-02
**API Version:** 1.0
