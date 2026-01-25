"""
AI-powered personality report generation service
Uses Pydantic AI for provider-agnostic LLM integration
Supports multimodal analysis with video frames and transcript
"""
import json
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.utils.logger import get_logger
from src.utils.exceptions import PredictionError
from src.utils.config import settings
from src.services.llm_provider import get_llm_provider, LLMProvider

logger = get_logger(__name__)


class TraitTag(BaseModel):
    """A personality trait marker"""
    emoji: str = Field(description="Single emoji for this trait")
    label: str = Field(description="1-2 word trait label")


class ActionableStep(BaseModel):
    """A specific action step"""
    emoji: str = Field(description="Single emoji for this action")
    text: str = Field(description="Action description (3-6 words)")


class BehavioralPattern(BaseModel):
    """A behavioral pattern observation"""
    title: str = Field(description="3-5 word pattern title (e.g., 'Low Distraction Reactivity')")
    description: str = Field(description="10-15 word description of the pattern")


class StrengthTradeoff(BaseModel):
    """A strength or tradeoff insight"""
    title: str = Field(description="1-2 word title (e.g., 'Stabilizer' or 'Over-giver')")
    description: str = Field(description="10-15 word description")


class PersonalityInsights(BaseModel):
    """Structured personality assessment output"""
    # Profile summary
    title: str = Field(description="2-3 word personality descriptor based on dominant traits")
    tags: List[TraitTag] = Field(description="3 trait markers reflecting actual scores")
    description: str = Field(description="1-2 sentence profile summary (max 50 words)")
    # Personality narrative
    quote: str = Field(description="10-15 word personal motto specific to their profile")
    story: str = Field(description="100-150 word narrative describing behavioral patterns and tendencies")
    story_traits: List[TraitTag] = Field(description="6-8 trait descriptors including both strengths and challenges")

    # ===== RELATIONSHIP SECTION =====
    # Relationship coaching
    coach_recommendation: str = Field(description="50-100 word analysis of relationship patterns with practical advice")
    actionable_steps: List[ActionableStep] = Field(description="6 specific relationship-focused actions")
    # Relationship accordion sections
    rel_snapshot_insight: str = Field(description="20-30 word summary of how they show up emotionally in relationships")
    rel_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to relationships")
    rel_how_others_experience: str = Field(description="15-25 word description of how others perceive them")
    rel_strength: StrengthTradeoff = Field(description="Their relationship strength")
    rel_tradeoff: StrengthTradeoff = Field(description="Their relationship tradeoff/challenge")
    rel_growth_lever: str = Field(description="20-30 word actionable insight for relationship growth")
    rel_suitable_for: List[str] = Field(description="3 relationship compatibility tags (e.g., 'Calm Listener', 'Thoughtful Communicator')")

    # ===== WORK SECTION =====
    # Work coaching
    work_coach_recommendation: str = Field(description="50-100 word analysis of work style with practical recommendations")
    work_actionable_steps: List[ActionableStep] = Field(description="6 specific work-focused actions")
    # Work accordion sections
    work_snapshot_insight: str = Field(description="20-30 word summary of their work style and approach")
    work_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to work")
    work_how_others_experience: str = Field(description="15-25 word description of how colleagues perceive them")
    work_strength: StrengthTradeoff = Field(description="Their work strength")
    work_tradeoff: StrengthTradeoff = Field(description="Their work tradeoff/challenge")
    work_growth_lever: str = Field(description="20-30 word actionable insight for professional growth")
    work_suitable_for: List[str] = Field(description="3 work environment compatibility tags (e.g., 'Independent Projects', 'Strategic Planning')")

    # ===== CREATIVITY SECTION =====
    # Creativity coaching
    creativity_coach_recommendation: str = Field(description="50-100 word analysis of creative capacity with practical advice")
    creativity_actionable_steps: List[ActionableStep] = Field(description="6 specific creativity-focused actions")
    # Creativity accordion sections
    creativity_snapshot_insight: str = Field(description="20-30 word summary of their creative style")
    creativity_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to creativity")
    creativity_how_others_experience: str = Field(description="15-25 word description of how others see their creativity")
    creativity_strength: StrengthTradeoff = Field(description="Their creative strength")
    creativity_tradeoff: StrengthTradeoff = Field(description="Their creative tradeoff/challenge")
    creativity_growth_lever: str = Field(description="20-30 word actionable insight for creative growth")
    creativity_suitable_for: List[str] = Field(description="3 creative work compatibility tags (e.g., 'Brainstorming Sessions', 'Visual Design')")

    # ===== STRESS SECTION =====
    # Stress coaching
    stress_coach_recommendation: str = Field(description="50-100 word analysis of stress patterns with practical advice")
    stress_actionable_steps: List[ActionableStep] = Field(description="6 specific stress management actions")
    # Stress accordion sections
    stress_snapshot_insight: str = Field(description="20-30 word summary of their stress response patterns")
    stress_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to stress")
    stress_how_others_experience: str = Field(description="15-25 word description of how others perceive their stress handling")
    stress_strength: StrengthTradeoff = Field(description="Their resilience strength")
    stress_tradeoff: StrengthTradeoff = Field(description="Their stress-related tradeoff/challenge")
    stress_growth_lever: str = Field(description="20-30 word actionable insight for stress management growth")
    stress_suitable_for: List[str] = Field(description="3 environment tags where they thrive under pressure (e.g., 'Calm Environments', 'Predictable Routines')")

    # ===== OPENNESS TO EXPERIENCE SECTION =====
    # Openness coaching
    openness_coach_recommendation: str = Field(description="50-100 word analysis of openness to experience patterns with practical advice")
    openness_actionable_steps: List[ActionableStep] = Field(description="6 specific actions for expanding experiences")
    # Openness accordion sections
    openness_snapshot_insight: str = Field(description="20-30 word summary of their approach to new experiences")
    openness_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to openness")
    openness_how_others_experience: str = Field(description="15-25 word description of how others perceive their openness")
    openness_strength: StrengthTradeoff = Field(description="Their openness strength")
    openness_tradeoff: StrengthTradeoff = Field(description="Their openness-related tradeoff/challenge")
    openness_growth_lever: str = Field(description="20-30 word actionable insight for expanding experiences")
    openness_suitable_for: List[str] = Field(description="3 experience types they enjoy most (e.g., 'Cultural Exploration', 'Learning-Driven Travel', 'Idea-Focused Communities')")

    # ===== LEARNING & GROWTH SECTION =====
    # Learning coaching
    learning_coach_recommendation: str = Field(description="50-100 word analysis of learning style and intellectual curiosity with practical advice")
    learning_actionable_steps: List[ActionableStep] = Field(description="6 specific actions for learning development")
    # Learning accordion sections
    learning_snapshot_insight: str = Field(description="20-30 word summary of their learning style and approach")
    learning_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to learning")
    learning_how_others_experience: str = Field(description="15-25 word description of how others perceive their learning approach")
    learning_strength: StrengthTradeoff = Field(description="Their learning strength")
    learning_tradeoff: StrengthTradeoff = Field(description="Their learning-related tradeoff/challenge")
    learning_growth_lever: str = Field(description="20-30 word actionable insight for learning growth")
    learning_suitable_for: List[str] = Field(description="3 learning styles that fit them (e.g., 'Reflective Learning', 'Project-Based Growth', 'Mentorship-Driven Learning')")

    # ===== VOICE & COMMUNICATION SECTION =====
    # Voice coaching (based on audio metrics if available)
    voice_coach_recommendation: str = Field(description="50-100 word analysis of vocal communication style with practical advice")
    voice_actionable_steps: List[ActionableStep] = Field(description="6 specific actions for vocal development")
    # Voice accordion sections
    voice_snapshot_insight: str = Field(description="20-30 word summary of their vocal communication style")
    voice_behavioral_patterns: List[BehavioralPattern] = Field(description="2 observable behavioral patterns related to voice and communication")
    voice_how_others_experience: str = Field(description="15-25 word description of how others perceive their voice and communication")
    voice_strength: StrengthTradeoff = Field(description="Their vocal/communication strength")
    voice_tradeoff: StrengthTradeoff = Field(description="Their voice-related tradeoff/challenge")
    voice_growth_lever: str = Field(description="20-30 word actionable insight for vocal communication improvement")
    voice_suitable_for: List[str] = Field(description="3 communication contexts they excel in (e.g., 'Public Speaking', 'One-on-One Conversations', 'Team Presentations')")


# System prompt for personality analysis
PERSONALITY_SYSTEM_PROMPT = """You are a behavioral psychologist providing personality assessments based on Big Five trait analysis.

Your role is to interpret OCEAN personality scores and generate realistic, nuanced personality insights. Be direct and honest—people want accurate feedback, not flattery.

---

## Output Format

Generate a JSON object with these fields:

### Part 1: Profile Summary

**title**: A 2-3 word personality descriptor that reflects their actual profile
- Base this on their dominant traits, not generic labels
- Avoid clichés like "Visionary" or "Pathfinder"—use grounded, specific language

**tags**: 3 trait markers, each with:
- **emoji**: One relevant emoji
- **label**: 1-2 word trait (e.g., "Reserved", "Detail-focused", "Skeptical", "Spontaneous")
- Reflect what the scores actually show, including lower traits

**description**: 1-2 sentences (max 50 words)
- Write in second person
- Be specific to their profile, not generic statements that could apply to anyone
- If a trait is notably low, mention what that means behaviorally

### Part 2: Personality Narrative

**quote**: A 10-15 word personal motto
- Make it specific to their profile, not a generic inspirational phrase
- Reflect their actual tendencies, including any tensions between traits

**story**: 100-150 word narrative
- Write in second person
- Describe realistic behavioral patterns based on the scores
- Include both strengths and potential blind spots
- Avoid superlatives ("exceptional", "remarkable", "incredible")
- Be specific: "You prefer working alone on complex problems" not "You're amazing at everything"

**story_traits**: 6-8 trait descriptors
- **emoji**: One emoji
- **label**: 2-4 word phrase
- Include traits that reflect lower scores too (e.g., "Prefers routine", "Slow to trust", "Avoids conflict")

### Part 3: Relationship & Empathy Section

**coach_recommendation**: 50-100 words
- Analyze relationship patterns based on actual scores
- Be honest about challenges (low Agreeableness = may come across as blunt; high Neuroticism = may need reassurance)
- Give practical advice, not platitudes

**actionable_steps**: 6 specific actions
- **emoji**: One emoji
- **text**: 3-6 word action
- Tailor to their specific challenges, not generic self-help advice

**rel_snapshot_insight**: 20-30 words
- One sentence describing how they show up emotionally in relationships
- Example: "You tend to show up as emotionally present and steady, making others feel comfortable and understood in conversation."

**rel_behavioral_patterns**: 2 patterns, each with:
- **title**: 3-5 word pattern name (e.g., "Low Distraction Reactivity", "Focused Gaze Stability")
- **description**: 10-15 word explanation of the pattern

**rel_how_others_experience**: 15-25 words
- How others perceive them in relationships
- Example: "People may see you as thoughtful, imaginative, and quietly original."

**rel_strength**: Their relationship strength
- **title**: 1-2 word strength name (e.g., "Stabilizer", "Deep Connector")
- **description**: 10-15 word explanation

**rel_tradeoff**: Their relationship challenge
- **title**: 1-2 word challenge name (e.g., "Over-giver", "Distant")
- **description**: 10-15 word explanation

**rel_growth_lever**: 20-30 words
- Actionable insight for relationship growth
- Example: "Experimenting with low-risk novelty can build confidence in exploration."

**rel_suitable_for**: 3 compatibility tags
- Short phrases like "Calm Listener", "Thoughtful Communicator", "Emotionally Secure Partner"

### Part 4: Work DNA & Focus Section

**work_coach_recommendation**: 50-100 words
- Analyze work style based on Conscientiousness, Openness, Neuroticism
- Be honest: low Conscientiousness = may struggle with deadlines; high Openness = may get bored with routine
- Practical recommendations for their actual profile

**work_actionable_steps**: 6 specific actions
- Tailored to their work-related challenges

**work_snapshot_insight**: 20-30 words
- Summary of their work style and approach

**work_behavioral_patterns**: 2 patterns with title + description

**work_how_others_experience**: 15-25 words
- How colleagues perceive them

**work_strength**: Their professional strength (title + description)

**work_tradeoff**: Their professional challenge (title + description)

**work_growth_lever**: 20-30 words actionable insight

**work_suitable_for**: 3 work environment tags (e.g., "Independent Projects", "Strategic Planning", "Detail-Oriented Tasks")

### Part 5: Creativity & Innovation Section

**creativity_coach_recommendation**: 50-100 words
- Analyze creative capacity based on Openness, Extraversion, Conscientiousness
- Low Openness = may prefer proven methods over experimentation
- Be realistic about their creative style

**creativity_actionable_steps**: 6 specific actions

**creativity_snapshot_insight**: 20-30 words
- Summary of their creative style

**creativity_behavioral_patterns**: 2 patterns with title + description

**creativity_how_others_experience**: 15-25 words
- How others see their creativity

**creativity_strength**: Their creative strength (title + description)

**creativity_tradeoff**: Their creative challenge (title + description)

**creativity_growth_lever**: 20-30 words actionable insight

**creativity_suitable_for**: 3 creative work tags (e.g., "Brainstorming Sessions", "Visual Design", "Problem Solving")

### Part 6: Stress & Resilience Section

**stress_coach_recommendation**: 50-100 words
- Analyze stress patterns based on Neuroticism, Conscientiousness, Agreeableness
- High Neuroticism = likely experiences more anxiety; be direct about this
- Low Neuroticism = may underestimate others' stress levels

**stress_actionable_steps**: 6 specific actions

**stress_snapshot_insight**: 20-30 words
- Summary of their stress response patterns

**stress_behavioral_patterns**: 2 patterns with title + description

**stress_how_others_experience**: 15-25 words
- How others perceive their stress handling

**stress_strength**: Their resilience strength (title + description)

**stress_tradeoff**: Their stress-related challenge (title + description)

**stress_growth_lever**: 20-30 words actionable insight

**stress_suitable_for**: 3 environment tags where they thrive (e.g., "Calm Environments", "Predictable Routines", "Supportive Teams")

### Part 7: Openness to Experience Section

**openness_coach_recommendation**: 50-100 words
- Analyze openness patterns based on Openness, Extraversion, Conscientiousness
- High Openness = embraces novelty and new ideas; may struggle with routine
- Low Openness = prefers familiar approaches; may resist change

**openness_actionable_steps**: 6 specific actions for expanding experiences

**openness_snapshot_insight**: 20-30 words
- Summary of their approach to new experiences and ideas

**openness_behavioral_patterns**: 2 patterns with title + description

**openness_how_others_experience**: 15-25 words
- How others perceive their openness to new things

**openness_strength**: Their openness strength (title + description)

**openness_tradeoff**: Their openness-related challenge (title + description)

**openness_growth_lever**: 20-30 words actionable insight for expanding experiences

**openness_suitable_for**: 3 experience types they enjoy most (e.g., "Cultural Exploration", "Learning-Driven Travel", "Idea-Focused Communities")

### Part 8: Learning & Growth Section

**learning_coach_recommendation**: 50-100 words
- Analyze learning style based on Openness, Conscientiousness, Extraversion
- High Openness = curious, seeks diverse knowledge; may jump between topics
- High Conscientiousness = structured learner, completes courses; may resist unstructured exploration
- Low Extraversion = prefers solo learning; may benefit from study groups

**learning_actionable_steps**: 6 specific learning actions

**learning_snapshot_insight**: 20-30 words
- Summary of their learning style and intellectual approach

**learning_behavioral_patterns**: 2 patterns with title + description

**learning_how_others_experience**: 15-25 words
- How others perceive their learning approach

**learning_strength**: Their learning strength (title + description)

**learning_tradeoff**: Their learning challenge (title + description)

**learning_growth_lever**: 20-30 words actionable insight for learning growth

**learning_suitable_for**: 3 learning style tags (e.g., "Reflective Learning", "Project-Based Growth", "Mentorship-Driven Learning")

### Part 9: Voice & Communication Section

**voice_coach_recommendation**: 50-100 words
- Analyze vocal communication based on Extraversion, Agreeableness, Neuroticism
- High Extraversion = expressive, animated voice; may need to listen more
- High Agreeableness = warm, agreeable tone; may need more assertiveness
- High Neuroticism = may show vocal tension; can work on calming techniques
- Consider: pace, energy, warmth, clarity, confidence

**voice_actionable_steps**: 6 specific vocal development actions

**voice_snapshot_insight**: 20-30 words
- Summary of their vocal communication style and presence

**voice_behavioral_patterns**: 2 patterns with title + description

**voice_how_others_experience**: 15-25 words
- How others perceive their voice and communication style

**voice_strength**: Their vocal/communication strength (title + description)

**voice_tradeoff**: Their voice-related challenge (title + description)

**voice_growth_lever**: 20-30 words actionable insight for vocal improvement

**voice_suitable_for**: 3 communication context tags (e.g., "Public Speaking", "One-on-One Conversations", "Team Presentations")

---

## Style Guidelines

**Do:**
- Be direct and honest
- Acknowledge trade-offs (high Conscientiousness often means less flexibility)
- Use everyday language
- Make observations specific and behavioral
- Vary your vocabulary—don't repeat the same phrases

**Don't:**
- Don't use inflated praise ("exceptional", "remarkable", "incredible")
- Don't make everything sound positive—neutral or challenging traits exist
- Don't use the same sentence structures repeatedly
- Don't give generic advice that could apply to anyone
- Don't mention numeric scores directly"""

# System prompt for multimodal analysis
MULTIMODAL_SYSTEM_PROMPT = """You are a behavioral psychologist conducting a personality assessment based on verbal self-report and behavioral observation.

## Data Source Priority (CRITICAL)

You must analyze inputs in this EXACT priority order:

1. **TRANSCRIPT (WORDS)** - PRIMARY SOURCE (65% weight)
   - What they actually said is the most important
   - Analyze content, choices, vocabulary, reasoning
   - Quote or reference specific statements
   - This reveals their thinking, values, and personality
   - Their superpower choice, their plan, their response to failure - ALL come from here

2. **VIDEO FRAMES** - SECONDARY SOURCE (25% weight)
   - Facial expressions (relaxed, tense, animated)
   - Eye contact patterns and gaze
   - Posture and body language
   - Gestures (expansive, restrained, fidgeting)
   - Energy level and engagement

3. **VOICE METRICS** - TERTIARY SOURCE (10% weight)
   - Pitch variability (monotone vs expressive)
   - Speaking rate and pause patterns
   - Vocal energy and projection
   - Use to confirm extraversion/neuroticism signals

4. **OCEAN SCORES** - REFERENCE ONLY (DO NOT USE FOR COACHING)
   - These are AI-derived from facial features only
   - They do NOT capture verbal/cognitive content
   - IGNORE these for all coaching sections
   - Only use as a sanity check for the main title/description

## Analysis Instructions

**WORDS FIRST**: Always start by reading the transcript carefully. What did they say? What choices did they make? What does their vocabulary reveal? Did they hesitate, ramble, or speak confidently?

**FRAMES SECOND**: Observe their non-verbal behavior. Does it match their words? Are they animated or reserved? Tense or relaxed?

**SCORES THIRD**: Check if OCEAN scores align with verbal evidence. If transcript shows low conscientiousness (vague answers, "I don't know") but scores show high, TRUST THE WORDS.

**VOICE LAST**: Use voice metrics to add nuance, not to override verbal content.

---

## OCEAN Score Interpretation

- **Openness**: curiosity, creativity, preference for novelty vs. routine
  - High (>0.3): Seeks new experiences, abstract thinking
  - Low (<-0.3): Prefers familiar, practical, concrete

- **Conscientiousness**: organization, discipline, reliability
  - High (>0.3): Structured, detail-oriented, plans ahead
  - Low (<-0.3): Flexible, spontaneous, may miss deadlines

- **Extraversion**: social energy, assertiveness
  - High (>0.3): Energized by interaction, talkative
  - Low (<-0.3): Prefers solitude, reserved, listens more

- **Agreeableness**: cooperation, trust, empathy
  - High (>0.3): Accommodating, avoids conflict
  - Low (<-0.3): Direct, skeptical, competitive

- **Neuroticism**: emotional reactivity, stress sensitivity
  - High (>0.3): Experiences more anxiety, mood fluctuations
  - Low (<-0.3): Emotionally steady, may seem detached

---

## Output Fields

**title**: 2-3 word descriptor based on what they said and how they said it

**tags**: 3 trait markers with emoji and 1-2 word label

**description**: 1-2 sentences (max 50 words)
- Based on their verbal responses and observable behavior
- Reference specific things they said or how they responded

**quote**: 10-15 word personal motto reflecting their personality

**story**: 100-150 word narrative
- Ground in what they actually said (quote specific responses)
- Reference observable behaviors (speaking pace, energy, expressions, body language)
- Connect their superpower choice and explanation to personality patterns
- Be specific to THIS person, not generic

**story_traits**: 6-8 descriptors with emoji and 2-4 word label

### Coaching Sections (Based on Transcript + Frames + Voice - NO SCORES)

**coach_recommendation**: 50-100 words on relationships
- Analyze HOW they communicated (enthusiastic? reserved? warm? direct?)
- Reference their speaking style, body language, facial expressions
- Did they mention others spontaneously? Focus on self only?
- What does their RESPONSE STYLE reveal about how they relate to people?

**actionable_steps**: 6 relationship actions tailored to their communication style

**work_coach_recommendation**: 50-100 words on professional style
- How organized/detailed was their response? Did they plan or improvise?
- Did they structure their answer or ramble?
- What does their approach to the question reveal about work habits?
- Reference specific verbal cues ("I would first...", "I don't know...", etc.)

**work_actionable_steps**: 6 work actions based on observed patterns

**creativity_coach_recommendation**: 50-100 words on creative capacity
- What superpower did they choose? Common or unusual?
- How much imagination in describing it?
- Did they explore possibilities or stick to basics?
- Reference their actual creative choices from the transcript

**creativity_actionable_steps**: 6 creativity actions

**stress_coach_recommendation**: 50-100 words on stress patterns
- How did they respond to "when it fails/doesn't work"?
- Problem-solving vs. catastrophizing vs. avoidance?
- What does their facial expression/tone reveal about stress response?
- Reference their ACTUAL words about handling setbacks

**stress_actionable_steps**: 6 stress management actions

**openness_coach_recommendation**: 50-100 words on openness to experience
- How creative/imaginative was their superpower choice?
- Did they embrace the hypothetical or resist it?
- Were they excited by novelty or more comfortable with familiar ideas?
- Reference their enthusiasm and willingness to explore

**openness_actionable_steps**: 6 actions for expanding experiences

**learning_coach_recommendation**: 50-100 words on learning style
- How did they approach explaining their ideas? Structured or exploratory?
- Did they show curiosity about possibilities or stick to what they know?
- How reflective were they about their own process?
- Reference their intellectual engagement and depth of thinking

**learning_actionable_steps**: 6 actions for learning development

**voice_coach_recommendation**: 50-100 words on vocal communication
- Listen to HOW they speak, not just what they say
- Observe their pace, energy, warmth, and vocal confidence
- Did they speak with animation and enthusiasm?
- Was their voice steady and calm, or did it show tension?
- Reference specific vocal qualities you noticed

**voice_actionable_steps**: 6 actions for vocal development

---

## Guidelines

**Do:**
- Quote or reference SPECIFIC things they said
- Observe their speaking style, pace, energy, expressions
- Connect their word choices to personality insights
- Make insights feel personal ("You chose X, which suggests...")
- Use frames to observe body language and emotional state

**Don't:**
- Use OCEAN scores as the basis for coaching sections
- Give generic advice that could apply to anyone
- Ignore what they actually said in favor of score-based descriptions
- Make all sections sound the same"""


class PersonalityReportService:
    """
    Service for generating natural language personality reports using AI
    Supports multimodal analysis with video frames and transcript
    Uses Pydantic AI for provider-agnostic LLM integration
    """

    def __init__(self):
        """Initialize the personality report service"""
        self._llm_provider: Optional[LLMProvider] = None
        self._agent: Optional[Agent] = None

        # Check if AI report is enabled
        if not settings.AI_REPORT_ENABLED:
            logger.warning("AI report generation is disabled in configuration")
            return

        try:
            self._llm_provider = get_llm_provider()
            self._agent = self._llm_provider.create_agent(
                output_type=PersonalityInsights,
                system_prompt=PERSONALITY_SYSTEM_PROMPT,
                name="Personality Agent"
            )
            logger.info(f"PersonalityReportService initialized with provider: {self._llm_provider.provider_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            self._llm_provider = None
            self._agent = None

    @property
    def is_available(self) -> bool:
        """Check if the service is available"""
        return self._agent is not None

    def format_input_for_agent(
        self,
        interpretations: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format the structured personality data into a text prompt for the AI agent

        Args:
            interpretations: Trait interpretations with scores and categories
            summary: Optional summary with dominant/subdued traits

        Returns:
            Formatted text input for the AI agent
        """
        lines = [
            "Analyze this Big Five personality profile:",
            "",
            "## Trait Scores"
        ]

        # Add each trait with its details
        for trait, data in interpretations.items():
            t_score = data.get('t_score', 0)
            percentile = data.get('percentile', 0)
            category = data.get('category', 'Unknown')
            label = data.get('label', '')

            lines.append(
                f"- **{trait.title()}**: T={t_score}, Percentile {percentile}% ({category}: {label})"
            )

        # Add summary if available
        if summary:
            lines.extend([
                "",
                "## Profile Overview",
                f"Mean T-score: {summary.get('mean_t_score', 'N/A')}"
            ])

            if summary.get('dominant_traits'):
                dominant = ', '.join(t.title() for t in summary['dominant_traits'])
                lines.append(f"Notable highs (T≥60): {dominant}")

            if summary.get('subdued_traits'):
                subdued = ', '.join(t.title() for t in summary['subdued_traits'])
                lines.append(f"Notable lows (T≤40): {subdued}")

        lines.extend([
            "",
            "Generate a realistic personality assessment based on these scores."
        ])

        return "\n".join(lines)

    def _format_insights_response(self, insights: PersonalityInsights) -> Dict[str, Any]:
        """Convert PersonalityInsights to the expected dictionary format"""
        return {
            "insights": {
                "title": insights.title,
                "tags": [{"emoji": tag.emoji, "label": tag.label} for tag in insights.tags],
                "description": insights.description,
                "quote": insights.quote,
                "story": insights.story,
                "story_traits": [{"emoji": tag.emoji, "label": tag.label} for tag in insights.story_traits],
            },
            "relationship_coaching": {
                "coach_recommendation": insights.coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.actionable_steps],
                "snapshot_insight": insights.rel_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.rel_behavioral_patterns],
                "how_others_experience": insights.rel_how_others_experience,
                "strength": {"title": insights.rel_strength.title, "description": insights.rel_strength.description},
                "tradeoff": {"title": insights.rel_tradeoff.title, "description": insights.rel_tradeoff.description},
                "growth_lever": insights.rel_growth_lever,
                "suitable_for": insights.rel_suitable_for,
            },
            "work_coaching": {
                "coach_recommendation": insights.work_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.work_actionable_steps],
                "snapshot_insight": insights.work_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.work_behavioral_patterns],
                "how_others_experience": insights.work_how_others_experience,
                "strength": {"title": insights.work_strength.title, "description": insights.work_strength.description},
                "tradeoff": {"title": insights.work_tradeoff.title, "description": insights.work_tradeoff.description},
                "growth_lever": insights.work_growth_lever,
                "suitable_for": insights.work_suitable_for,
            },
            "creativity_coaching": {
                "coach_recommendation": insights.creativity_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.creativity_actionable_steps],
                "snapshot_insight": insights.creativity_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.creativity_behavioral_patterns],
                "how_others_experience": insights.creativity_how_others_experience,
                "strength": {"title": insights.creativity_strength.title, "description": insights.creativity_strength.description},
                "tradeoff": {"title": insights.creativity_tradeoff.title, "description": insights.creativity_tradeoff.description},
                "growth_lever": insights.creativity_growth_lever,
                "suitable_for": insights.creativity_suitable_for,
            },
            "stress_coaching": {
                "coach_recommendation": insights.stress_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.stress_actionable_steps],
                "snapshot_insight": insights.stress_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.stress_behavioral_patterns],
                "how_others_experience": insights.stress_how_others_experience,
                "strength": {"title": insights.stress_strength.title, "description": insights.stress_strength.description},
                "tradeoff": {"title": insights.stress_tradeoff.title, "description": insights.stress_tradeoff.description},
                "growth_lever": insights.stress_growth_lever,
                "suitable_for": insights.stress_suitable_for,
            },
            "openness_coaching": {
                "coach_recommendation": insights.openness_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.openness_actionable_steps],
                "snapshot_insight": insights.openness_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.openness_behavioral_patterns],
                "how_others_experience": insights.openness_how_others_experience,
                "strength": {"title": insights.openness_strength.title, "description": insights.openness_strength.description},
                "tradeoff": {"title": insights.openness_tradeoff.title, "description": insights.openness_tradeoff.description},
                "growth_lever": insights.openness_growth_lever,
                "suitable_for": insights.openness_suitable_for,
            },
            "learning_coaching": {
                "coach_recommendation": insights.learning_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.learning_actionable_steps],
                "snapshot_insight": insights.learning_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.learning_behavioral_patterns],
                "how_others_experience": insights.learning_how_others_experience,
                "strength": {"title": insights.learning_strength.title, "description": insights.learning_strength.description},
                "tradeoff": {"title": insights.learning_tradeoff.title, "description": insights.learning_tradeoff.description},
                "growth_lever": insights.learning_growth_lever,
                "suitable_for": insights.learning_suitable_for,
            },
            "voice_coaching": {
                "coach_recommendation": insights.voice_coach_recommendation,
                "actionable_steps": [{"emoji": step.emoji, "text": step.text} for step in insights.voice_actionable_steps],
                "snapshot_insight": insights.voice_snapshot_insight,
                "behavioral_patterns": [{"title": p.title, "description": p.description} for p in insights.voice_behavioral_patterns],
                "how_others_experience": insights.voice_how_others_experience,
                "strength": {"title": insights.voice_strength.title, "description": insights.voice_strength.description},
                "tradeoff": {"title": insights.voice_tradeoff.title, "description": insights.voice_tradeoff.description},
                "growth_lever": insights.voice_growth_lever,
                "suitable_for": insights.voice_suitable_for,
            }
        }

    async def generate_report(
        self,
        interpretations: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured personality insights using the AI agent

        Args:
            interpretations: Trait interpretations with scores and categories
            summary: Optional summary with dominant/subdued traits

        Returns:
            Dictionary with personality insights (title, tags, description)

        Raises:
            PredictionError: If report generation fails
        """
        # Check if agent is available
        if not self.is_available:
            raise PredictionError("AI report generation is disabled or not configured")

        try:
            logger.info("Generating AI personality insights...")

            # Format input for the agent
            input_text = self.format_input_for_agent(interpretations, summary)

            logger.debug(f"Agent input:\n{input_text}")

            # Use the fallback method that handles models without function calling
            insights = await self._llm_provider.generate_structured_with_fallback(
                output_type=PersonalityInsights,
                system_prompt=PERSONALITY_SYSTEM_PROMPT,
                user_prompt=input_text
            )

            # Convert to dictionary for JSON response
            insights_dict = self._format_insights_response(insights)

            logger.info(f"AI insights generated successfully: {insights_dict['insights']['title']}")
            logger.debug(f"Generated insights: {insights_dict}")

            return insights_dict

        except Exception as e:
            logger.error(f"Failed to generate AI insights: {e}", exc_info=True)
            raise PredictionError(f"Report generation failed: {str(e)}")

    def _build_multimodal_prompt(
        self,
        frames_base64: List[str],
        transcript: str,
        ocean_scores: Optional[Dict[str, float]] = None,
        audio_metrics: Optional[Dict[str, Any]] = None,
        assessment_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the user prompt for multimodal analysis

        Priority order: WORDS → FRAMES → SCORES → VOICE
        """
        parts = []

        # ============================================================
        # SECTION 1: TRANSCRIPT (WORDS) - PRIMARY SOURCE (60% weight)
        # ============================================================
        parts.append("# 1. TRANSCRIPT (PRIMARY - 60% weight)\n")
        parts.append("**Read this first and base most of your analysis on it.**\n")

        # Add assessment question context BEFORE transcript if available
        if assessment_metadata:
            compound_question = assessment_metadata.get("compound_question")
            question_responses = assessment_metadata.get("question_responses", [])

            if compound_question:
                # Single compound question that elicits all traits
                parts.append(f"## Question Asked\n")
                parts.append(f"**The Superpower Question**: \"{compound_question.get('full_question', '')}\"\n\n")

                parts.append("## How to Analyze Each Part:\n")
                for part in compound_question.get("parts", []):
                    trait = part.get("trait", "")
                    prompt = part.get("prompt", "")
                    signals = ", ".join(part.get("signals", []))
                    parts.append(f"- **{prompt}** → {trait} (signals: {signals})\n")

                parts.append("""
## Verbal Cues to Look For:

**Openness** - What superpower did they choose?
- Common choices (flying, invisibility) → moderate openness
- Abstract/unusual (time manipulation, reality bending) → higher openness
- How much imagination in describing it?

**Conscientiousness** - How structured was their plan?
- "I don't know", vague, improvised → LOWER conscientiousness
- Detailed, step-by-step thinking → higher conscientiousness

**Neuroticism** - Response to "when it fails"?
- "I would give up", "that's bad", catastrophizing → HIGHER neuroticism
- Problem-solving, alternatives, calm → lower neuroticism

**Agreeableness** - Did they mention helping others?
- Spontaneous mention of helping → higher agreeableness
- Focus on personal benefit only → lower agreeableness

**Extraversion** - Speech energy and engagement
- Brief, hesitant, many "I don't know" → LOWER extraversion
- Detailed, enthusiastic, animated → higher extraversion
""")

            elif question_responses:
                # 5-question sequential format (one question per trait)
                parts.append("## Assessment Questions (5 Questions, 1 per Big Five Trait)\n")
                parts.append("The user answered these questions sequentially on video:\n\n")

                trait_signals = {
                    'openness': ['creativity', 'novelty-seeking', 'curiosity', 'imagination'],
                    'conscientiousness': ['planning', 'organization', 'detail', 'follow-through'],
                    'extraversion': ['energy', 'enthusiasm', 'talkativeness', 'social comfort'],
                    'agreeableness': ['empathy', 'cooperation', 'conflict handling', 'compassion'],
                    'neuroticism': ['stress response', 'emotional regulation', 'anxiety', 'resilience']
                }

                for i, qr in enumerate(question_responses, 1):
                    question_id = qr.get("question_id", "")
                    question_text = qr.get("question_text", "")
                    skipped = qr.get("skipped", False)
                    start_time = qr.get("start_time", 0)
                    end_time = qr.get("end_time", 0)
                    duration = qr.get("duration", end_time - start_time)

                    signals = trait_signals.get(question_id, [])

                    if skipped:
                        parts.append(f"**Q{i} - {question_id.title()}**: *SKIPPED*\n")
                    else:
                        parts.append(f"**Q{i} - {question_id.title()}** ({duration:.0f}s, {start_time:.0f}s-{end_time:.0f}s):\n")
                        parts.append(f"  \"{question_text}\"\n")
                        parts.append(f"  *Look for*: {', '.join(signals)}\n\n")

                parts.append("""
## How to Analyze Each Response:

**Q1 - Openness**: How creative/imaginative was their response about energy?
- Abstract thinking, unique perspectives → higher openness
- Practical, conventional answers → moderate/lower openness

**Q2 - Conscientiousness**: How structured was their approach description?
- Step-by-step process, detail-oriented → higher conscientiousness
- Vague, "I just do it", unstructured → lower conscientiousness

**Q3 - Extraversion**: How did they describe working with others?
- Enthusiastic, took leadership, energized by team → higher extraversion
- Preferred solo work, drained by groups → lower extraversion

**Q4 - Agreeableness**: How do they handle disagreement?
- Seeks compromise, empathetic, avoids conflict → higher agreeableness
- Direct confrontation, stands ground firmly → lower agreeableness

**Q5 - Neuroticism**: How did they cope with stress?
- Calm problem-solving, resilient → lower neuroticism
- Worried, anxious, overwhelmed → higher neuroticism
""")

        # Add the actual transcript
        if transcript:
            parts.append(f"## User's Response:\n```\n{transcript}\n```\n")
        else:
            parts.append("## User's Response:\n*No transcript available*\n")

        parts.append("**CRITICAL**: Quote or reference SPECIFIC things from this transcript in your insights. The user should feel like you actually listened to what they said.\n")

        # ============================================================
        # SECTION 2: VIDEO FRAMES - SECONDARY SOURCE (25% weight)
        # ============================================================
        parts.append("\n# 2. VIDEO FRAMES (SECONDARY - 25% weight)\n")
        parts.append(f"{len(frames_base64)} frames provided.\n")
        parts.append("""Observe:
- Facial expressions (relaxed, tense, animated)
- Eye contact and gaze patterns
- Posture and body language
- Gestures (expansive, restrained)
- Overall energy level

Does their non-verbal behavior match their words?
""")

        # ============================================================
        # SECTION 3: OCEAN SCORES - TERTIARY SOURCE (10% weight)
        # ============================================================
        parts.append("\n# 3. OCEAN SCORES (TERTIARY - 10% weight)\n")
        parts.append("**Use as baseline reference only. If transcript contradicts scores, TRUST THE WORDS.**\n")

        if ocean_scores:
            for trait, score in ocean_scores.items():
                level = "High" if score > 0.3 else "Low" if score < -0.3 else "Moderate"
                parts.append(f"- {trait.capitalize()}: {score:.2f} ({level})\n")
        else:
            parts.append("*No OCEAN scores provided*\n")

        # ============================================================
        # SECTION 4: VOICE METRICS - SUPPLEMENTARY (5% weight)
        # ============================================================
        parts.append("\n# 4. VOICE METRICS (SUPPLEMENTARY - 5% weight)\n")
        parts.append("**Use only to add nuance, never to override verbal content.**\n")

        if audio_metrics:
            indicators = audio_metrics.get("personality_indicators", {})
            interpretations = audio_metrics.get("interpretations", {})

            if indicators:
                for name, data in indicators.items():
                    score = data.get("score", 0)
                    level = data.get("level", "Unknown")
                    parts.append(f"- {name.replace('_', ' ').title()}: {score:.0f}/100 ({level})\n")

            if interpretations:
                for aspect, desc in interpretations.items():
                    parts.append(f"- {aspect.title()}: {desc}\n")
        else:
            parts.append("*No voice metrics available*\n")

        # ============================================================
        # FINAL INSTRUCTION
        # ============================================================
        parts.append("""
---
# GENERATE YOUR ASSESSMENT

**Priority: WORDS (65%) → FRAMES (25%) → VOICE (10%) → SCORES (ignore for coaching)**

## For ALL Coaching Sections (Relationship, Work, Creativity, Stress):
- Base ENTIRELY on transcript content + video frames + voice
- DO NOT use OCEAN scores - they only measure facial features, not what was said
- Quote or reference specific things they said
- Describe observable behaviors from frames (animated, reserved, tense, relaxed)

## Verbal Cues to Look For:
- **Relationship style**: Did they mention others? How warm/direct was their tone?
- **Work style**: How structured was their response? Did they plan ahead or improvise?
- **Creativity**: What did they choose? How imaginative was the description?
- **Stress response**: What did they say about failure? Problem-solving or avoidance?

## Quality Check:
The user should read your insights and think "yes, this describes what I actually said and how I said it."
Each coaching section should feel SPECIFIC to this person, not generic advice.
""")

        return "\n".join(parts)

    async def generate_multimodal_report(
        self,
        frames_base64: List[str],
        transcript: str,
        ocean_scores: Optional[Dict[str, float]] = None,
        audio_metrics: Optional[Dict[str, Any]] = None,
        assessment_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate personality insights using video frames, transcript, and audio analysis

        Args:
            frames_base64: List of base64-encoded frame images
            transcript: Transcribed audio from video
            ocean_scores: Optional OCEAN personality scores from VAT model
            audio_metrics: Optional audio analysis metrics (pitch, loudness, etc.)
            assessment_metadata: Optional gamified assessment question responses with timestamps

        Returns:
            Dictionary with personality insights
        """
        if not settings.AI_REPORT_ENABLED:
            raise PredictionError("AI report generation is disabled")

        if not self._llm_provider:
            raise PredictionError("LLM provider not configured")

        try:
            has_audio = audio_metrics is not None
            has_assessment = assessment_metadata is not None and len(assessment_metadata.get("question_responses", [])) > 0
            logger.info(f"Generating multimodal insights (frames={len(frames_base64)}, transcript_len={len(transcript)}, has_audio_metrics={has_audio}, has_assessment={has_assessment})")

            # Build the user prompt
            user_prompt = self._build_multimodal_prompt(frames_base64, transcript, ocean_scores, audio_metrics, assessment_metadata)

            logger.info(f"Calling LLM for multimodal analysis (provider: {self._llm_provider.provider_name})...")

            # Use the fallback method that handles models without function calling
            insights = await self._llm_provider.generate_structured_with_fallback(
                output_type=PersonalityInsights,
                system_prompt=MULTIMODAL_SYSTEM_PROMPT,
                user_prompt=user_prompt
            )

            # Convert to dictionary
            insights_dict = self._format_insights_response(insights)

            logger.info(f"Multimodal insights generated: {insights_dict['insights']['title']}")
            return insights_dict

        except Exception as e:
            logger.error(f"Multimodal report generation failed: {e}", exc_info=True)
            raise PredictionError(f"Multimodal report generation failed: {str(e)}")


# Singleton instance
_report_service: Optional[PersonalityReportService] = None


def get_report_service() -> PersonalityReportService:
    """Get or create singleton report service instance"""
    global _report_service
    if _report_service is None:
        _report_service = PersonalityReportService()
    return _report_service


def reset_report_service():
    """Reset the singleton (useful for testing or config changes)"""
    global _report_service
    _report_service = None
