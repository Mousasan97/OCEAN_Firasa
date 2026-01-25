"""
AI Personality Coach Agent using PydanticAI

Provides conversational AI coaching based on the user's OCEAN personality analysis.
Uses PydanticAI for structured agent definition with tools and dependencies.
Supports both synchronous and streaming responses.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from src.services.llm_provider import get_llm_provider
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PersonalityContext:
    """
    Dependencies for the personality coach agent.
    Contains the user's personality analysis data.
    """
    # OCEAN scores (0-1 scale)
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    # Derived metrics
    derived_metrics: Optional[Dict[str, Any]] = None

    # Trait interpretations with descriptions
    interpretations: Optional[Dict[str, Any]] = None

    # User's spoken responses from video recording
    user_transcript: Optional[str] = None

    # Full analysis report (if available)
    full_report: Optional[Dict[str, Any]] = None

    # Conversation history
    message_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    """Response model for the personality coach."""
    response: str
    suggestions: Optional[List[str]] = None


# =============================================================================
# System Prompts
# =============================================================================

PERSONALITY_COACH_SYSTEM_PROMPT = """You are Firasa, an empathetic personality coach for OCEAN personality analysis.

## Response Style
- Be CONCISE: 2-3 short paragraphs maximum
- Use bullet points for tips (max 3-4 tips)
- Speak naturally and warmly
- Reference the user's EXACT scores and personalized insights from the tools

## Score Interpretation (0-100%)
- 0-35%: Low
- 35-65%: Moderate
- 65-100%: High

## OCEAN Traits
- **Openness**: Creativity, curiosity, new experiences
- **Conscientiousness**: Organization, discipline, goals
- **Extraversion**: Social energy, assertiveness
- **Agreeableness**: Cooperation, empathy, trust
- **Neuroticism**: Emotional sensitivity (lower = calmer)

## Tool Usage
IMPORTANT: When the user asks about a specific trait or wants to improve:
1. Use get_trait_interpretation to get their PERSONALIZED analysis conclusion for that trait
2. Use get_ocean_scores for an overview of all scores
3. Use get_all_interpretations to see all personalized insights
4. Use get_user_responses to see what the user SAID during their video recording - this gives you context about their perspective and communication style

The interpretation tools contain the user's actual analysis results with personalized descriptions. The user transcript shows what they actually said during assessment questions - use this to give more relevant, contextualized advice. No trait is "good" or "bad"."""


# =============================================================================
# Agent Definition
# =============================================================================

def create_personality_coach() -> Agent[PersonalityContext, str]:
    """
    Create the personality coach agent with tools and system prompt.

    Returns:
        PydanticAI Agent configured for personality coaching
    """
    llm_provider = get_llm_provider()

    agent = Agent(
        llm_provider.model,
        deps_type=PersonalityContext,
        output_type=str,
        system_prompt=PERSONALITY_COACH_SYSTEM_PROMPT
    )

    # Register tools
    @agent.tool
    def get_ocean_scores(ctx: RunContext[PersonalityContext]) -> str:
        """Get the user's OCEAN personality scores with interpretations."""
        scores = ctx.deps

        def interpret_level(score: float) -> str:
            pct = score * 100
            if pct < 35:
                return "Low"
            elif pct < 65:
                return "Moderate"
            else:
                return "High"

        def fmt(score: float) -> str:
            return f"{round(score * 100)}%"

        return f"""User's OCEAN Scores:
- Openness: {fmt(scores.openness)} ({interpret_level(scores.openness)})
- Conscientiousness: {fmt(scores.conscientiousness)} ({interpret_level(scores.conscientiousness)})
- Extraversion: {fmt(scores.extraversion)} ({interpret_level(scores.extraversion)})
- Agreeableness: {fmt(scores.agreeableness)} ({interpret_level(scores.agreeableness)})
- Neuroticism: {fmt(scores.neuroticism)} ({interpret_level(scores.neuroticism)})"""

    @agent.tool
    def get_trait_details(ctx: RunContext[PersonalityContext], trait_name: str) -> str:
        """
        Get detailed information about a specific OCEAN trait.

        Args:
            trait_name: One of 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
        """
        scores = ctx.deps
        trait_name = trait_name.lower()

        trait_info = {
            'openness': {
                'score': scores.openness,
                'high_desc': 'You are imaginative, curious, and open to new ideas. You enjoy exploring abstract concepts and appreciate art, adventure, and variety.',
                'low_desc': 'You prefer familiarity and practical approaches. You value tradition and tend to be more conventional in your thinking.',
                'improve_tips': [
                    'Try a new hobby or activity outside your comfort zone each month',
                    'Read books from genres you normally avoid',
                    'Travel to unfamiliar places when possible',
                    'Engage in creative activities like art, music, or writing'
                ]
            },
            'conscientiousness': {
                'score': scores.conscientiousness,
                'high_desc': 'You are organized, disciplined, and goal-oriented. You plan ahead and follow through on commitments.',
                'low_desc': 'You are spontaneous and flexible, preferring to go with the flow rather than stick to rigid plans.',
                'improve_tips': [
                    'Use a planner or digital calendar consistently',
                    'Break large goals into smaller, actionable tasks',
                    'Create routines for important daily activities',
                    'Set deadlines for yourself, even when not required'
                ]
            },
            'extraversion': {
                'score': scores.extraversion,
                'high_desc': 'You are outgoing, energetic, and thrive in social situations. You enjoy being around others and often take initiative.',
                'low_desc': 'You are more introverted, preferring deeper one-on-one connections or solitary activities. You recharge through alone time.',
                'improve_tips': [
                    'Join groups or clubs aligned with your interests',
                    'Practice starting conversations with new people',
                    'Volunteer for leadership opportunities',
                    'Schedule regular social activities, even small ones'
                ]
            },
            'agreeableness': {
                'score': scores.agreeableness,
                'high_desc': 'You are cooperative, empathetic, and prioritize harmony in relationships. You tend to trust others and are willing to help.',
                'low_desc': 'You are more competitive and skeptical. You prioritize your own interests and question others\' motives.',
                'improve_tips': [
                    'Practice active listening in conversations',
                    'Look for opportunities to help others without expecting return',
                    'Try to understand different perspectives before judging',
                    'Express appreciation and gratitude more often'
                ]
            },
            'neuroticism': {
                'score': scores.neuroticism,
                'high_desc': 'You experience emotions intensely and may be more sensitive to stress. This can also mean deep emotional awareness.',
                'low_desc': 'You are emotionally stable and resilient. You handle stress well and maintain a calm demeanor.',
                'improve_tips': [
                    'Develop a regular mindfulness or meditation practice',
                    'Exercise regularly to manage stress hormones',
                    'Keep a journal to process emotions',
                    'Learn cognitive reframing techniques for negative thoughts'
                ]
            }
        }

        if trait_name not in trait_info:
            return f"Unknown trait: {trait_name}. Valid traits are: openness, conscientiousness, extraversion, agreeableness, neuroticism"

        info = trait_info[trait_name]
        score = info['score']
        pct = round(score * 100)
        level = "High" if pct >= 65 else "Moderate" if pct >= 35 else "Low"
        desc = info['high_desc'] if pct >= 50 else info['low_desc']

        tips = "\n".join(f"• {tip}" for tip in info['improve_tips'][:3])  # Limit to 3 tips

        return f"""{trait_name.title()}: {pct}% ({level})

{desc}

Quick Tips:
{tips}"""

    @agent.tool
    def get_derived_metrics(ctx: RunContext[PersonalityContext]) -> str:
        """Get the user's derived personality metrics (stress resilience, creativity, etc.)."""
        if not ctx.deps.derived_metrics:
            return "Derived metrics not available for this analysis."

        metrics = ctx.deps.derived_metrics
        result_lines = ["Derived Personality Metrics:"]

        for key, value in metrics.items():
            if isinstance(value, dict):
                score = value.get('score', value.get('percentage', 'N/A'))
                level = value.get('level', '')
                result_lines.append(f"- {key}: {score}% ({level})")
            else:
                result_lines.append(f"- {key}: {value}")

        return "\n".join(result_lines)

    @agent.tool
    def get_personality_strengths(ctx: RunContext[PersonalityContext]) -> str:
        """Identify the user's top personality strengths based on their OCEAN profile."""
        scores = ctx.deps

        traits = [
            ('Openness', scores.openness),
            ('Conscientiousness', scores.conscientiousness),
            ('Extraversion', scores.extraversion),
            ('Agreeableness', scores.agreeableness),
            ('Emotional Stability', 1 - scores.neuroticism)  # Invert for positive framing
        ]

        # Sort by score descending
        sorted_traits = sorted(traits, key=lambda x: x[1], reverse=True)

        strengths = []
        for trait, score in sorted_traits[:3]:
            if score >= 0.5:
                strengths.append(f"• {trait} ({score:.1%})")

        if not strengths:
            # If no high scores, still show top 2
            strengths = [f"• {t[0]} ({t[1]:.1%})" for t in sorted_traits[:2]]

        return f"""Top Personality Strengths:
{chr(10).join(strengths)}

These traits represent your natural tendencies and can be leveraged for personal and professional success."""

    @agent.tool
    def get_growth_areas(ctx: RunContext[PersonalityContext]) -> str:
        """Identify potential growth areas based on the user's OCEAN profile."""
        scores = ctx.deps

        traits = [
            ('Openness', scores.openness),
            ('Conscientiousness', scores.conscientiousness),
            ('Extraversion', scores.extraversion),
            ('Agreeableness', scores.agreeableness),
            ('Emotional Stability', 1 - scores.neuroticism)
        ]

        # Sort by score ascending
        sorted_traits = sorted(traits, key=lambda x: x[1])

        growth_areas = []
        for trait, score in sorted_traits[:2]:
            if score < 0.5:
                growth_areas.append(f"• {trait} ({score:.1%})")

        if not growth_areas:
            return "Your personality profile is well-balanced! Focus on leveraging your existing strengths."

        return f"""Potential Growth Areas:
{chr(10).join(growth_areas)}

Remember: These are opportunities for development, not weaknesses. Growth in these areas can complement your existing strengths."""

    @agent.tool
    def get_trait_interpretation(ctx: RunContext[PersonalityContext], trait_name: str) -> str:
        """
        Get the personalized interpretation and insights for a specific trait.
        This includes the analysis conclusion and personalized description.

        Args:
            trait_name: One of 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
        """
        if not ctx.deps.interpretations:
            return f"No detailed interpretation available for {trait_name}."

        trait_name = trait_name.lower()
        interp = ctx.deps.interpretations.get(trait_name, {})

        if not interp:
            return f"No interpretation data found for {trait_name}."

        t_score = interp.get('t_score', 50)
        level = interp.get('level', 'Moderate')
        interpretation = interp.get('interpretation', '')
        raw_score = interp.get('raw_score', 0.5)

        result = f"""{trait_name.title()}: {t_score}% ({level})

Analysis Conclusion:
{interpretation}"""

        return result

    @agent.tool
    def get_all_interpretations(ctx: RunContext[PersonalityContext]) -> str:
        """Get all trait interpretations with personalized insights from the analysis."""
        if not ctx.deps.interpretations:
            return "No detailed interpretations available."

        trait_order = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        results = ["Personalized Trait Analysis:\n"]

        for trait in trait_order:
            interp = ctx.deps.interpretations.get(trait, {})
            if interp:
                t_score = interp.get('t_score', 50)
                level = interp.get('level', 'Moderate')
                interpretation = interp.get('interpretation', 'No interpretation available.')

                results.append(f"**{trait.title()}** ({t_score}% - {level}):")
                results.append(f"{interpretation}\n")

        return "\n".join(results)

    @agent.tool
    def get_user_responses(ctx: RunContext[PersonalityContext]) -> str:
        """
        Get the user's spoken responses from their video recording.
        This shows what the user said during the personality assessment questions.
        Use this to understand the user's perspective and give more personalized advice.
        """
        if not ctx.deps.user_transcript:
            return "No transcript available from the user's recording."

        return f"""User's Spoken Responses (from video recording):

{ctx.deps.user_transcript}

Use these responses to understand the user's perspective, communication style, and give more personalized, relevant advice."""

    return agent


# =============================================================================
# Service Class
# =============================================================================

class PersonalityCoachService:
    """
    Service for managing personality coach conversations.
    """

    def __init__(self):
        self._agent: Optional[Agent[PersonalityContext, str]] = None

    @property
    def agent(self) -> Agent[PersonalityContext, str]:
        """Lazy-load the agent."""
        if self._agent is None:
            self._agent = create_personality_coach()
        return self._agent

    async def chat(
        self,
        message: str,
        ocean_scores: Dict[str, float],
        derived_metrics: Optional[Dict[str, Any]] = None,
        interpretations: Optional[Dict[str, Any]] = None,
        user_transcript: Optional[str] = None,
        full_report: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Send a message to the personality coach and get a response.

        Args:
            message: User's message
            ocean_scores: Dict with keys 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
            derived_metrics: Optional derived metrics from analysis
            interpretations: Optional trait interpretations with descriptions
            user_transcript: Optional user's spoken responses from video recording
            full_report: Optional full analysis report
            message_history: Optional list of previous messages [{'role': 'user'/'assistant', 'content': '...'}]

        Returns:
            Assistant's response string
        """
        # Build context
        context = PersonalityContext(
            openness=ocean_scores.get('openness', 0.5),
            conscientiousness=ocean_scores.get('conscientiousness', 0.5),
            extraversion=ocean_scores.get('extraversion', 0.5),
            agreeableness=ocean_scores.get('agreeableness', 0.5),
            neuroticism=ocean_scores.get('neuroticism', 0.5),
            derived_metrics=derived_metrics,
            interpretations=interpretations,
            user_transcript=user_transcript,
            full_report=full_report,
            message_history=message_history
        )

        try:
            # Build message history for multi-turn conversation
            if message_history:
                from pydantic_ai import ModelRequest, ModelResponse, UserPromptPart, TextPart

                messages = []
                for msg in message_history:
                    if msg['role'] == 'user':
                        messages.append(ModelRequest(parts=[UserPromptPart(content=msg['content'])]))
                    else:
                        messages.append(ModelResponse(parts=[TextPart(content=msg['content'])]))

                result = await self.agent.run(message, deps=context, message_history=messages)
            else:
                result = await self.agent.run(message, deps=context)

            logger.info(f"Personality coach responded to: {message[:50]}...")
            return result.output

        except Exception as e:
            logger.error(f"Error in personality coach chat: {e}")
            raise

    async def chat_stream(
        self,
        message: str,
        ocean_scores: Dict[str, float],
        derived_metrics: Optional[Dict[str, Any]] = None,
        interpretations: Optional[Dict[str, Any]] = None,
        user_transcript: Optional[str] = None,
        full_report: Optional[Dict[str, Any]] = None,
        message_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[str]:
        """
        Send a message to the personality coach and stream the response.

        Args:
            message: User's message
            ocean_scores: Dict with keys 'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'
            derived_metrics: Optional derived metrics from analysis
            interpretations: Optional trait interpretations with descriptions
            user_transcript: Optional user's spoken responses from video recording
            full_report: Optional full analysis report
            message_history: Optional list of previous messages [{'role': 'user'/'assistant', 'content': '...'}]

        Yields:
            Text chunks as they are generated (deltas)
        """
        # Build context
        context = PersonalityContext(
            openness=ocean_scores.get('openness', 0.5),
            conscientiousness=ocean_scores.get('conscientiousness', 0.5),
            extraversion=ocean_scores.get('extraversion', 0.5),
            agreeableness=ocean_scores.get('agreeableness', 0.5),
            neuroticism=ocean_scores.get('neuroticism', 0.5),
            derived_metrics=derived_metrics,
            interpretations=interpretations,
            user_transcript=user_transcript,
            full_report=full_report,
            message_history=message_history
        )

        try:
            # Build message history for multi-turn conversation
            messages = None
            if message_history:
                from pydantic_ai import ModelRequest, ModelResponse, UserPromptPart, TextPart

                messages = []
                for msg in message_history:
                    if msg['role'] == 'user':
                        messages.append(ModelRequest(parts=[UserPromptPart(content=msg['content'])]))
                    else:
                        messages.append(ModelResponse(parts=[TextPart(content=msg['content'])]))

            # Use run_stream for streaming response
            async with self.agent.run_stream(
                message,
                deps=context,
                message_history=messages
            ) as result:
                # Stream text as deltas (incremental chunks)
                async for chunk in result.stream_text(delta=True):
                    yield chunk

            logger.info(f"Personality coach streamed response to: {message[:50]}...")

        except Exception as e:
            logger.error(f"Error in personality coach stream: {e}")
            raise


# Singleton instance
_coach_service: Optional[PersonalityCoachService] = None


def get_personality_coach() -> PersonalityCoachService:
    """Get or create singleton personality coach service."""
    global _coach_service
    if _coach_service is None:
        _coach_service = PersonalityCoachService()
    return _coach_service
