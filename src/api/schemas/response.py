"""
API response schemas
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class QuestionResponse(BaseModel):
    """A single question response from the gamified assessment"""
    question_id: str = Field(description="Question identifier (openness, conscientiousness, extraversion, agreeableness, neuroticism)")
    question_text: str = Field(description="The question that was asked")
    skipped: bool = Field(default=False, description="Whether the question was skipped")
    start_time: float = Field(description="Start time in seconds into the combined video")
    end_time: float = Field(description="End time in seconds into the combined video")
    duration: float = Field(description="Duration of the response in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "question_id": "openness",
                "question_text": "If you had a superpower, what would it be and how would you use it creatively in your daily life?",
                "skipped": False,
                "start_time": 0.0,
                "end_time": 38.5,
                "duration": 38.5
            }
        }


class AssessmentMetadata(BaseModel):
    """Metadata for the gamified personality assessment"""
    question_responses: List[QuestionResponse] = Field(description="List of question responses with timestamps")
    total_questions: int = Field(default=5, description="Total number of questions in assessment")
    questions_answered: int = Field(description="Number of questions answered (not skipped)")
    questions_skipped: int = Field(description="Number of questions skipped")

    class Config:
        json_schema_extra = {
            "example": {
                "question_responses": [
                    {
                        "question_id": "openness",
                        "question_text": "If you had a superpower...",
                        "skipped": False,
                        "start_time": 0.0,
                        "end_time": 38.5,
                        "duration": 38.5
                    }
                ],
                "total_questions": 5,
                "questions_answered": 4,
                "questions_skipped": 1
            }
        }


class PersonalityTraits(BaseModel):
    """Personality trait scores"""

    extraversion: float = Field(description="Outgoing, social, energetic")
    neuroticism: float = Field(description="Anxious, moody, emotionally unstable")
    agreeableness: float = Field(description="Cooperative, trusting, helpful")
    conscientiousness: float = Field(description="Organized, disciplined, responsible")
    openness: float = Field(description="Creative, curious, open to new experiences")

    class Config:
        json_schema_extra = {
            "example": {
                "extraversion": 0.65,
                "neuroticism": -0.23,
                "agreeableness": 0.45,
                "conscientiousness": 0.78,
                "openness": 0.34
            }
        }


class TraitTag(BaseModel):
    """A personality trait tag with emoji"""
    emoji: str = Field(description="Emoji representing the trait")
    label: str = Field(description="Short trait label")


class InsightActionableStep(BaseModel):
    """An actionable development step in insights"""
    emoji: str = Field(description="Emoji for the step")
    text: str = Field(description="Step description")


class PersonalityInsights(BaseModel):
    """AI-generated personality insights - overview only (coaching content is in metrics sections)"""
    # Top insights card fields
    title: str = Field(description="Creative personality title (e.g., 'Visionary Pathfinder')")
    tags: List[TraitTag] = Field(description="3 trait tags with emojis")
    description: str = Field(description="Brief personality description")
    # Personality story fields
    quote: str = Field(description="Poetic personality tagline")
    story: str = Field(description="Detailed personality narrative paragraph")
    story_traits: List[TraitTag] = Field(description="6-8 'You are' trait descriptors")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Visionary Pathfinder",
                "tags": [
                    {"emoji": "🤝", "label": "Empathetic"},
                    {"emoji": "🌿", "label": "Calm"},
                    {"emoji": "🧠", "label": "Emotionally aware"}
                ],
                "description": "Your facial cues suggest a natural openness and curiosity, often seen in individuals who enjoy exploring new ideas and connecting with others.",
                "quote": "A heart that listens and eyes that roam—curious, kind, and ever at home.",
                "story": "You move through life with a gentle curiosity that draws others to you. Your natural warmth creates safe spaces for meaningful conversations.",
                "story_traits": [
                    {"emoji": "🎯", "label": "Purpose-driven"},
                    {"emoji": "💭", "label": "Deep thinker"},
                    {"emoji": "🌊", "label": "Emotionally fluid"}
                ]
            }
        }


class DerivedMetric(BaseModel):
    """A single derived personality metric"""
    score: float = Field(description="Score as percentage (0-100)")
    level: str = Field(description="Level: Low, Moderate, or High")
    description: str = Field(description="Description of the score meaning")


class ActionableStep(BaseModel):
    """An actionable development step"""
    emoji: str = Field(description="Emoji for the step")
    text: str = Field(description="Step description")


class BehavioralPattern(BaseModel):
    """A behavioral pattern observation"""
    title: str = Field(description="Pattern title")
    description: str = Field(description="Pattern description")


class StrengthTradeoff(BaseModel):
    """A strength or tradeoff insight"""
    title: str = Field(description="Short title")
    description: str = Field(description="Description")


class RelationshipMetrics(BaseModel):
    """Relationships and empathy metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (trust_signaling, social_openness, empathic_disposition, conflict_avoidance, harmony_seeking, anxiety_avoidance)")
    coach_recommendation: str = Field(description="Personalized coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for development")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of how they show up emotionally")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Observable behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How others perceive them")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Relationship strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Relationship challenge")
    growth_lever: Optional[str] = Field(default=None, description="Actionable growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Compatibility tags")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "trust_signaling": {
                        "score": 89,
                        "level": "High",
                        "description": "Your warmth and stability naturally inspire trust in others."
                    },
                    "social_openness": {
                        "score": 81,
                        "level": "High",
                        "description": "You thrive in social settings and actively seek new connections."
                    },
                    "empathic_disposition": {
                        "score": 57,
                        "level": "Moderate",
                        "description": "You balance empathy with objectivity in understanding others."
                    },
                    "conflict_avoidance": {
                        "score": 25,
                        "level": "Low",
                        "description": "You're comfortable with direct confrontation and asserting boundaries."
                    },
                    "harmony_seeking": {
                        "score": 45,
                        "level": "Moderate",
                        "description": "You value harmony but can be direct when necessary."
                    },
                    "anxiety_avoidance": {
                        "score": 30,
                        "level": "Low",
                        "description": "You handle confrontation with emotional stability and composure."
                    }
                },
                "coach_recommendation": "Your natural warmth and reliability are powerful assets in building rapport. Practice active listening and mirroring subtle positive expressions to deepen your connections.",
                "actionable_steps": [
                    {"emoji": "🤝", "text": "Join social or online communities"},
                    {"emoji": "🌱", "text": "Start conversations with new people"},
                    {"emoji": "👂", "text": "Practice active listening"}
                ]
            }
        }


class WorkMetrics(BaseModel):
    """Work DNA and focus metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (persistence, focus_attention, structure_preference, risk_aversion)")
    coach_recommendation: str = Field(description="Personalized work coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for professional development")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of work style")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Work behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How colleagues perceive them")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Professional strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Professional challenge")
    growth_lever: Optional[str] = Field(default=None, description="Professional growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Work environment tags")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "persistence": {
                        "score": 78,
                        "level": "High",
                        "description": "You demonstrate strong determination and follow-through on commitments."
                    },
                    "focus_attention": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You can maintain focus reasonably well but may benefit from structured work environments."
                    },
                    "structure_preference": {
                        "score": 55,
                        "level": "Moderate",
                        "description": "You appreciate some structure but also value flexibility when needed."
                    },
                    "risk_aversion": {
                        "score": 35,
                        "level": "Low",
                        "description": "You're comfortable taking calculated risks and may seek out challenging opportunities."
                    }
                },
                "coach_recommendation": "Your exceptional determination and follow-through are valuable assets in the workplace. You have the discipline to see complex projects through to completion.",
                "actionable_steps": [
                    {"emoji": "🎯", "text": "Set small daily goals"},
                    {"emoji": "⏰", "text": "Use Pomodoro technique"},
                    {"emoji": "📝", "text": "Document risk assessments"}
                ]
            }
        }


class CreativityMetrics(BaseModel):
    """Creativity pulse metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (ideation_power, openness_to_novelty, originality_index, attention_to_detail_creative)")
    coach_recommendation: str = Field(description="Personalized creativity coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for creative development")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of creative style")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Creative behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How others see their creativity")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Creative strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Creative challenge")
    growth_lever: Optional[str] = Field(default=None, description="Creative growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Creative work tags")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "ideation_power": {
                        "score": 78,
                        "level": "High",
                        "description": "Your high capacity for ideation is a great strength! You naturally generate innovative ideas."
                    },
                    "openness_to_novelty": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You balance appreciation for the familiar with openness to new experiences."
                    },
                    "originality_index": {
                        "score": 55,
                        "level": "Moderate",
                        "description": "You balance originality with practicality in your creative work."
                    },
                    "attention_to_detail_creative": {
                        "score": 30,
                        "level": "Low",
                        "description": "You thrive in the initial ideation phase but may benefit from collaboration for refinement."
                    }
                },
                "coach_recommendation": "Your high capacity for ideation is a great strength! Try brainstorming techniques or mind mapping to capture and develop your unique ideas.",
                "actionable_steps": [
                    {"emoji": "🧠", "text": "Practice daily brainstorming"},
                    {"emoji": "🗺️", "text": "Use mind mapping techniques"},
                    {"emoji": "🤝", "text": "Partner with detail-oriented people"}
                ]
            }
        }


class StressMetrics(BaseModel):
    """Stress and resilience metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (stress_indicators, emotional_regulation, resilience_score)")
    coach_recommendation: str = Field(description="Personalized stress and resilience coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for stress management and resilience building")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of stress response style")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Stress behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How others perceive their stress handling")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Resilience strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Resilience challenge")
    growth_lever: Optional[str] = Field(default=None, description="Resilience growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Environment suitability tags")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "stress_indicators": {
                        "score": 35,
                        "level": "Low",
                        "description": "You tend to remain calm under pressure and handle stressors effectively."
                    },
                    "emotional_regulation": {
                        "score": 72,
                        "level": "High",
                        "description": "You excel at managing your emotional responses and maintaining composure."
                    },
                    "resilience_score": {
                        "score": 68,
                        "level": "Moderate",
                        "description": "You have reasonable resilience but may benefit from strengthening your coping toolkit."
                    }
                },
                "coach_recommendation": "Your natural calmness and emotional stability are valuable assets for handling life's challenges. You demonstrate strong resilience and recover well from setbacks.",
                "actionable_steps": [
                    {"emoji": "🧘", "text": "Practice daily meditation"},
                    {"emoji": "🏃", "text": "Exercise regularly"},
                    {"emoji": "🌐", "text": "Build a support network"}
                ]
            }
        }


class OpennessMetrics(BaseModel):
    """Openness to experience metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (openness_to_experience, novelty_seeking, risk_tolerance_adventure, planning_preference)")
    coach_recommendation: str = Field(description="Personalized openness coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for expanding openness to experience")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of openness style")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Openness behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How others perceive their openness")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Openness strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Openness challenge")
    growth_lever: Optional[str] = Field(default=None, description="Openness growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Experience suitability tags")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "openness_to_experience": {
                        "score": 78,
                        "level": "High",
                        "description": "You have exceptional curiosity and openness to new experiences, ideas, and perspectives."
                    },
                    "novelty_seeking": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You enjoy novelty in moderation, balancing exploration with comfort."
                    },
                    "risk_tolerance_adventure": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You take measured risks when the potential reward is clear."
                    },
                    "planning_preference": {
                        "score": 30,
                        "level": "Low",
                        "description": "You prefer spontaneity and flexibility over rigid plans and schedules."
                    }
                },
                "coach_recommendation": "Your exceptional curiosity and openness to ideas are valuable assets for growth and exploration! Your creativity flows best with flexibility.",
                "actionable_steps": [
                    {"emoji": "🌍", "text": "Explore different cultural perspectives"},
                    {"emoji": "🚀", "text": "Pursue ambitious new projects"},
                    {"emoji": "📝", "text": "Try light planning frameworks"}
                ],
                "suitable_for": ["Cultural Exploration", "Learning-Driven Travel", "Idea-Focused Communities"]
            }
        }


class LearningMetrics(BaseModel):
    """Learning and growth metrics derived from OCEAN scores"""
    metrics: Dict[str, DerivedMetric] = Field(description="Derived metrics (intellectual_curiosity, reflective_tendency, structured_learning_preference, adaptability_index)")
    coach_recommendation: str = Field(description="Personalized learning coach recommendation")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for learning development")
    # Accordion section fields
    snapshot_insight: Optional[str] = Field(default=None, description="Summary of learning style")
    behavioral_patterns: Optional[List[BehavioralPattern]] = Field(default=None, description="Learning behavioral patterns")
    how_others_experience: Optional[str] = Field(default=None, description="How others perceive their learning approach")
    strength: Optional[StrengthTradeoff] = Field(default=None, description="Learning strength")
    tradeoff: Optional[StrengthTradeoff] = Field(default=None, description="Learning challenge")
    growth_lever: Optional[str] = Field(default=None, description="Learning growth insight")
    suitable_for: Optional[List[str]] = Field(default=None, description="Learning styles that fit them")

    class Config:
        json_schema_extra = {
            "example": {
                "metrics": {
                    "intellectual_curiosity": {
                        "score": 78,
                        "level": "High",
                        "description": "You have a strong drive to learn, explore ideas, and understand how things work."
                    },
                    "reflective_tendency": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You reflect on experiences when prompted, balancing action with contemplation."
                    },
                    "structured_learning_preference": {
                        "score": 65,
                        "level": "Moderate",
                        "description": "You appreciate structure but can adapt to different learning formats."
                    },
                    "adaptability_index": {
                        "score": 30,
                        "level": "Low",
                        "description": "You prefer consistency and may need extra time to adjust to new approaches."
                    }
                },
                "coach_recommendation": "Your strong intellectual curiosity and reflective nature make you a natural learner. Embrace diverse sources of information and be open to unconventional learning paths.",
                "actionable_steps": [
                    {"emoji": "🤝", "text": "Join social or online communities"},
                    {"emoji": "🌱", "text": "Start conversations with new people"},
                    {"emoji": "👂", "text": "Practice active listening"}
                ],
                "suitable_for": ["Reflective Learning", "Project-Based Growth", "Mentorship-Driven Learning"]
            }
        }


class VocalIndicator(BaseModel):
    """A vocal personality indicator derived from acoustic analysis"""
    score: float = Field(description="Score as percentage (0-100)")
    level: str = Field(description="Level: Low, Moderate, or High")
    signals: List[str] = Field(description="Specific vocal signals detected")


class AudioMetrics(BaseModel):
    """Voice and speech metrics derived from audio analysis"""
    indicators: Dict[str, VocalIndicator] = Field(
        description="Personality indicators derived from voice (vocal_extraversion, vocal_stability, vocal_confidence, vocal_warmth)"
    )
    interpretations: Dict[str, str] = Field(
        description="Human-readable interpretations of vocal characteristics"
    )
    coach_recommendation: str = Field(description="Personalized recommendation based on vocal patterns")
    actionable_steps: List[ActionableStep] = Field(description="Actionable steps for vocal development")

    class Config:
        json_schema_extra = {
            "example": {
                "indicators": {
                    "vocal_extraversion": {
                        "score": 72,
                        "level": "High",
                        "signals": ["projects voice confidently", "speaks rapidly with few pauses", "uses expressive intonation"]
                    },
                    "vocal_stability": {
                        "score": 65,
                        "level": "Moderate",
                        "signals": ["maintains consistent vocal quality", "controlled dynamic range"]
                    },
                    "vocal_confidence": {
                        "score": 68,
                        "level": "Moderate",
                        "signals": ["speaks with authority", "maintains steady speech flow"]
                    },
                    "vocal_warmth": {
                        "score": 55,
                        "level": "Moderate",
                        "signals": ["comfortable speaking pace"]
                    }
                },
                "interpretations": {
                    "pitch": "Moderate pitch, balanced vocal tone",
                    "expressiveness": "Highly expressive speech with wide pitch variation",
                    "volume": "Moderate volume, conversational tone",
                    "pace": "Fast-paced speech with few pauses, rapid communication",
                    "brightness": "Balanced voice clarity",
                    "stability": "Moderately stable voice"
                },
                "coach_recommendation": "Your voice projects confidence and energy. The expressive intonation engages listeners effectively. Consider varying your pace occasionally to emphasize key points.",
                "actionable_steps": [
                    {"emoji": "🎤", "text": "Practice vocal warm-ups before important calls"},
                    {"emoji": "⏸️", "text": "Use strategic pauses for emphasis"},
                    {"emoji": "🎯", "text": "Record yourself to identify patterns"}
                ]
            }
        }


class PreprocessingMetadata(BaseModel):
    """Preprocessing metadata for video processing"""

    # Video-specific fields (VAT model)
    total_frames: Optional[int] = Field(None, description="Total frames in video")
    fps: Optional[float] = Field(None, description="Video frame rate")
    width: Optional[int] = Field(None, description="Video width")
    height: Optional[int] = Field(None, description="Video height")
    duration_seconds: Optional[float] = Field(None, description="Video duration in seconds")
    num_frames_extracted: Optional[int] = Field(None, description="Number of frames extracted")
    extraction_method: Optional[str] = Field(None, description="Frame extraction method (e.g., k_segment)")
    frame_indices: Optional[List[int]] = Field(None, description="Indices of extracted frames")
    preprocessing_method: Optional[str] = Field(None, description="Preprocessing method used")
    file_type: Optional[str] = Field(None, description="File type (video)")
    file_path: Optional[str] = Field(None, description="File path (if applicable)")

    # Legacy image fields (kept for backwards compatibility)
    original_size: Optional[tuple] = Field(None, description="Original image size (width, height)")
    processed_size: Optional[tuple] = Field(None, description="Processed image size")
    method: Optional[str] = Field(None, description="Preprocessing method used")
    face_detected: Optional[bool] = Field(None, description="Whether face was detected")
    face_bbox: Optional[Dict[str, int]] = Field(None, description="Face bounding box coordinates")


class PredictionMetadata(BaseModel):
    """Prediction metadata"""

    model: Optional[str] = Field(None, description="Model type (vat)")
    num_frames: Optional[int] = Field(None, description="Number of frames used for prediction")
    tta_used: Optional[bool] = Field(None, description="Whether TTA was applied")
    device: Optional[str] = Field(None, description="Device used for inference")
    # Legacy field
    image_size: Optional[tuple] = Field(None, description="Input image size (legacy)")


class PredictionResponse(BaseModel):
    """Prediction response schema"""

    success: bool = Field(description="Whether prediction was successful")
    predictions: PersonalityTraits = Field(description="Personality trait predictions")
    interpretations: Optional[Dict[str, Any]] = Field(None, description="T-score interpretations and narratives")
    summary: Optional[Dict[str, Any]] = Field(None, description="Personality profile summary")
    insights: Optional[PersonalityInsights] = Field(None, description="AI-generated personality overview (title, tags, description, quote, story). Coaching content is in metrics sections.")
    relationship_metrics: Optional[RelationshipMetrics] = Field(None, description="Relationships and empathy metrics derived from OCEAN scores")
    work_metrics: Optional[WorkMetrics] = Field(None, description="Work DNA and focus metrics derived from OCEAN scores")
    creativity_metrics: Optional[CreativityMetrics] = Field(None, description="Creativity pulse metrics derived from OCEAN scores")
    stress_metrics: Optional[StressMetrics] = Field(None, description="Stress and resilience metrics derived from OCEAN scores")
    openness_metrics: Optional[OpennessMetrics] = Field(None, description="Openness to experience metrics derived from OCEAN scores")
    learning_metrics: Optional[LearningMetrics] = Field(None, description="Learning and growth metrics derived from OCEAN scores")
    audio_metrics: Optional[AudioMetrics] = Field(None, description="Voice and speech metrics derived from audio analysis")
    report_error: Optional[str] = Field(None, description="Error message if report generation failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": {
                    "openness": 0.65,
                    "conscientiousness": 0.78,
                    "extraversion": 0.45,
                    "agreeableness": 0.52,
                    "neuroticism": -0.23
                },
                "insights": {
                    "title": "Visionary Pathfinder",
                    "tags": [
                        {"emoji": "🤝", "label": "Empathetic"},
                        {"emoji": "🌿", "label": "Calm"},
                        {"emoji": "🧠", "label": "Emotionally aware"}
                    ],
                    "description": "Your facial cues suggest a natural openness and curiosity, often seen in individuals who enjoy exploring new ideas and connecting with others.",
                    "quote": "A heart that listens and eyes that roam—curious, kind, and ever at home.",
                    "story": "You move through life with a gentle curiosity that draws others to you. Your natural warmth creates safe spaces for meaningful conversations.",
                    "story_traits": [
                        {"emoji": "🎯", "label": "Purpose-driven"},
                        {"emoji": "💭", "label": "Deep thinker"},
                        {"emoji": "🌊", "label": "Emotionally fluid"}
                    ]
                },
                "relationship_metrics": {
                    "metrics": {
                        "trust_signaling": {"score": 89, "level": "High", "description": "Your warmth and stability naturally inspire trust in others."},
                        "social_openness": {"score": 81, "level": "High", "description": "You thrive in social settings and actively seek new connections."},
                        "empathic_disposition": {"score": 57, "level": "Moderate", "description": "You balance empathy with objectivity in understanding others."},
                        "conflict_avoidance": {"score": 25, "level": "Low", "description": "You're comfortable with direct confrontation and asserting boundaries."},
                        "harmony_seeking": {"score": 45, "level": "Moderate", "description": "You value harmony but can be direct when necessary."},
                        "anxiety_avoidance": {"score": 30, "level": "Low", "description": "You handle confrontation with emotional stability and composure."}
                    },
                    "coach_recommendation": "Your natural warmth and reliability are powerful assets in building rapport.",
                    "actionable_steps": [
                        {"emoji": "🤝", "text": "Join social or online communities"},
                        {"emoji": "👂", "text": "Practice active listening"}
                    ]
                },
                "work_metrics": {
                    "metrics": {
                        "persistence": {"score": 78, "level": "High", "description": "You demonstrate strong determination and follow-through on commitments."},
                        "focus_attention": {"score": 65, "level": "Moderate", "description": "You can maintain focus reasonably well but may benefit from structured work environments."},
                        "structure_preference": {"score": 55, "level": "Moderate", "description": "You appreciate some structure but also value flexibility when needed."},
                        "risk_aversion": {"score": 35, "level": "Low", "description": "You're comfortable taking calculated risks and may seek out challenging opportunities."}
                    },
                    "coach_recommendation": "Your exceptional determination and follow-through are valuable assets in the workplace.",
                    "actionable_steps": [
                        {"emoji": "🎯", "text": "Set small daily goals"},
                        {"emoji": "⏰", "text": "Use Pomodoro technique"}
                    ]
                },
                "creativity_metrics": {
                    "metrics": {
                        "ideation_power": {"score": 78, "level": "High", "description": "Your high capacity for ideation is a great strength!"},
                        "openness_to_novelty": {"score": 65, "level": "Moderate", "description": "You balance appreciation for the familiar with openness to new experiences."},
                        "originality_index": {"score": 55, "level": "Moderate", "description": "You balance originality with practicality in your creative work."},
                        "attention_to_detail_creative": {"score": 30, "level": "Low", "description": "You thrive in the initial ideation phase but may benefit from collaboration for refinement."}
                    },
                    "coach_recommendation": "Your high capacity for ideation is a great strength! Try brainstorming techniques or mind mapping.",
                    "actionable_steps": [
                        {"emoji": "🧠", "text": "Practice daily brainstorming"},
                        {"emoji": "🗺️", "text": "Use mind mapping techniques"}
                    ]
                },
                "stress_metrics": {
                    "metrics": {
                        "stress_indicators": {"score": 35, "level": "Low", "description": "You tend to remain calm under pressure and handle stressors effectively."},
                        "emotional_regulation": {"score": 72, "level": "High", "description": "You excel at managing your emotional responses and maintaining composure."},
                        "resilience_score": {"score": 68, "level": "Moderate", "description": "You have reasonable resilience but may benefit from strengthening your coping toolkit."}
                    },
                    "coach_recommendation": "Your natural calmness and emotional stability are valuable assets for handling life's challenges.",
                    "actionable_steps": [
                        {"emoji": "🧘", "text": "Practice daily meditation"},
                        {"emoji": "🌐", "text": "Build a support network"}
                    ]
                },
                "metadata": {
                    "preprocessing": {
                        "total_frames": 300,
                        "fps": 30.0,
                        "duration_seconds": 10.0,
                        "num_frames_extracted": 32,
                        "extraction_method": "k_segment"
                    },
                    "prediction": {
                        "model": "vat",
                        "num_frames": 32,
                        "tta_used": True,
                        "device": "cuda"
                    },
                    "multimodal": {
                        "frames_analyzed": 10,
                        "has_transcript": True,
                        "transcript_length": 1500
                    }
                },
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""

    success: bool = Field(description="Whether batch prediction was successful")
    predictions: List[PersonalityTraits] = Field(description="List of predictions")
    count: int = Field(description="Number of predictions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Batch metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response schema"""

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_type: Optional[str] = Field(None, description="Error type/category")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "No face detected in image",
                "error_type": "FaceDetectionError",
                "details": {},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""

    status: str = Field(description="Health status (healthy/unhealthy)")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "model_info": {
                    "backbone": "resnet18",
                    "device": "cuda"
                },
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class TraitDescriptionsResponse(BaseModel):
    """Personality trait descriptions response"""

    traits: Dict[str, str] = Field(description="Trait name to description mapping")

    class Config:
        json_schema_extra = {
            "example": {
                "traits": {
                    "extraversion": "Outgoing, social, energetic",
                    "neuroticism": "Anxious, moody, emotionally unstable",
                    "agreeableness": "Cooperative, trusting, helpful",
                    "conscientiousness": "Organized, disciplined, responsible",
                    "openness": "Creative, curious, open to new experiences"
                }
            }
        }
