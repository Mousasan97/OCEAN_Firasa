"""
AI Personality Coach Agent using PydanticAI

Provides conversational AI coaching based on the user's OCEAN personality analysis.
Uses PydanticAI for structured agent definition with tools and dependencies.
Supports both synchronous and streaming responses.

Job search uses a hybrid approach:
- AI writes natural text response about jobs (stays in conversation history)
- Backend appends JSON data for frontend to render job cards
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.services.llm_provider import get_llm_provider
from src.services.cultural_fit_service import get_cultural_fit_analysis
from src.services.job_search_service import search_jobs_for_agent, get_job_search_service
from src.utils.logger import get_logger
from src.utils.config import settings

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

    # Career profile from CV upload (if available)
    career_profile: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for the personality coach."""
    response: str
    suggestions: Optional[List[str]] = None


# Global storage for last job search results (for backend to append to response)
_last_job_search_results: Optional[Dict[str, Any]] = None


def get_last_job_search_results() -> Optional[Dict[str, Any]]:
    """Get the last job search results (if any) and clear them."""
    global _last_job_search_results
    results = _last_job_search_results
    _last_job_search_results = None
    return results


def store_job_search_results(results: Dict[str, Any]) -> None:
    """Store job search results for backend to append to response."""
    global _last_job_search_results
    _last_job_search_results = results
    logger.info(f"Stored {len(results.get('jobs', []))} job results for later retrieval")


def format_cv_context(career_profile: Optional[Dict[str, Any]]) -> str:
    """
    Format CV data as a context string to prepend to user messages.
    Returns empty string if no CV data.
    """
    if not career_profile:
        return ""

    parts = []
    role = career_profile.get("current_role") or career_profile.get("target_role")
    location = career_profile.get("location")
    years = career_profile.get("years_experience")
    skills = career_profile.get("key_skills", [])[:5]  # Top 5 skills

    if role:
        parts.append(role)
    if location:
        parts.append(location)
    if years:
        parts.append(f"{years}y exp")
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")

    if parts:
        return f"[CV: {' | '.join(parts)}]\n\n"
    return ""


# =============================================================================
# System Prompts
# =============================================================================

PERSONALITY_COACH_SYSTEM_PROMPT = """You are Firasa, an empathetic personality coach for OCEAN personality analysis.

## CRITICAL RULE - JOB SEARCH WITH CV DATA
**WHEN YOU SEE [CV: ...] AT THE START OF A MESSAGE AND THE USER ASKS ABOUT JOBS/MATCHING/CAREERS:**
1. YOU MUST IMMEDIATELY CALL search_matching_jobs - DO NOT ASK QUESTIONS
2. Extract a GENERIC job title from the CV role:
   - "Senior Data Engineer" → "Data Engineer"
   - "AI Research Engineer-Sustainable Energy" → "AI Engineer"
   - "Software Developer - Payment Systems" → "Software Developer"
3. Simplify the location: "Milan, Italy" → "Italy", "Bolzano-Italy" → "Italy"
4. Call search_matching_jobs(role="Data Engineer", location="Italy")

**TRIGGER PHRASES (when CV data is present, CALL THE TOOL):**
- "find jobs" / "job matching" / "find matching jobs"
- "match my personality" / "jobs that match"
- "career opportunities" / "job search"

**DO NOT ASK** for role or location if [CV: ...] data is visible - USE IT.

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
5. Use get_cultural_fit when the user asks about careers, workplace culture, job fit, or work environments - this matches their personality to 12 workplace culture types

The interpretation tools contain the user's actual analysis results with personalized descriptions. The user transcript shows what they actually said during assessment questions - use this to give more relevant, contextualized advice. No trait is "good" or "bad".

## Cultural Fit Analysis
When asked about career matches, workplace fit, or cultural preferences:
- Use get_cultural_fit to analyze which workplace cultures match their personality
- The tool returns top culture type matches with fit scores, strengths, and potential challenges
- Culture types include: Startup Disruptor, Tech Innovator, Corporate Enterprise, Creative Agency, Mission-Driven, Consulting, Remote/Distributed, Family Business, Research/Academic, Healthcare, Government, and Entrepreneurial
- Focus on the top 2-3 matches and explain WHY they fit based on the user's personality dimensions

## Job Search (NO CV data)
If user asks about jobs but NO [CV: ...] data is shown:
- If user provides role + location → search immediately
- If missing role or location → ask ONLY for the missing piece

After calling search_matching_jobs:
- Write a brief, friendly summary of the top jobs found (mention 2-3 top positions by title and company)
- Explain WHY these jobs match their personality profile based on culture fit
- The job cards will automatically appear in a sidebar panel"""


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

    @agent.tool
    def get_cultural_fit(ctx: RunContext[PersonalityContext]) -> str:
        """
        Get cultural fit analysis matching the user's personality to workplace culture types.
        Use this when the user asks about careers, workplace environments, job fit, or cultural preferences.
        Returns top matching culture types with fit scores, strengths, and potential challenges.
        """
        scores = ctx.deps
        ocean_scores = {
            'openness': scores.openness,
            'conscientiousness': scores.conscientiousness,
            'extraversion': scores.extraversion,
            'agreeableness': scores.agreeableness,
            'neuroticism': scores.neuroticism
        }

        return get_cultural_fit_analysis(ocean_scores, ctx.deps.derived_metrics)

    @agent.tool
    def get_career_profile(ctx: RunContext[PersonalityContext]) -> str:
        """
        Get the user's career profile from their uploaded CV.
        Returns role, skills, experience, industries, and location.
        Use this when discussing careers, job search, or giving professional advice.
        """
        if not ctx.deps.career_profile:
            return "No CV has been uploaded. The user can upload their CV for personalized career advice and job matching."

        cp = ctx.deps.career_profile
        parts = ["User's Career Profile (from CV):"]

        if cp.get("current_role"):
            parts.append(f"- Current Role: {cp['current_role']}")
        if cp.get("target_role"):
            parts.append(f"- Target Role: {cp['target_role']}")
        if cp.get("years_experience"):
            parts.append(f"- Experience: {cp['years_experience']} years")
        if cp.get("location"):
            parts.append(f"- Location: {cp['location']}")
        if cp.get("key_skills"):
            skills = cp['key_skills'][:10]  # Max 10 skills
            parts.append(f"- Key Skills: {', '.join(skills)}")
        if cp.get("industries"):
            parts.append(f"- Industries: {', '.join(cp['industries'])}")
        if cp.get("education_level"):
            parts.append(f"- Education: {cp['education_level']}")
        if cp.get("certifications"):
            parts.append(f"- Certifications: {', '.join(cp['certifications'])}")
        if cp.get("summary"):
            parts.append(f"\nSummary: {cp['summary']}")

        return "\n".join(parts)

    @agent.tool
    def search_matching_jobs(ctx: RunContext[PersonalityContext], role: str, location: str) -> str:
        """
        Search for real job listings. CALL THIS TOOL when user asks about jobs/matching.

        WHEN CV DATA EXISTS (message starts with [CV: ...]):
        - Extract GENERIC job title: "Senior Data Engineer" → "Data Engineer"
        - Extract location: "Milan, Italy" → "Italy"
        - Call immediately, DO NOT ask questions

        Examples:
        - CV shows "Senior Data Engineer, Milan, Italy" + user says "find jobs" → role="Data Engineer", location="Italy"
        - CV shows "Software Developer, London" + user says "match my personality" → role="Software Developer", location="UK"
        - No CV, user says "software jobs in Berlin" → role="Software Engineer", location="Berlin"

        Args:
            role: Generic job title extracted from CV or user input
            location: Country or city from CV or user input
        """
        import json as json_module

        if not settings.JOB_SEARCH_ENABLED or not settings.SERPAPI_KEY:
            return "Job search is not currently enabled. Please configure SERPAPI_KEY to enable this feature."

        # Auto-fill from CV if available and "auto" is specified
        career_profile = ctx.deps.career_profile
        original_role = role
        original_location = location

        if career_profile:
            if role == "auto" or not role:
                role = career_profile.get("current_role") or career_profile.get("target_role")
            if location == "auto" or not location:
                location = career_profile.get("location")

        # Validate we have required fields
        if not role:
            return "Please specify a job role/title to search for, or upload your CV first."
        if not location:
            return "Please specify a location to search in, or upload your CV with your location."

        try:
            # Get the job search service
            service = get_job_search_service()
            jobs = service.search_jobs(role, location, num_results=8)

            if not jobs:
                return f"No jobs found for '{role}' in '{location}'. Try a different search term or broader location."

            # Get user's top culture matches for scoring
            scores = ctx.deps
            ocean_scores = {
                'openness': scores.openness,
                'conscientiousness': scores.conscientiousness,
                'extraversion': scores.extraversion,
                'agreeableness': scores.agreeableness,
                'neuroticism': scores.neuroticism
            }

            # Get cultural fit service for scoring
            from src.services.cultural_fit_service import CulturalFitService
            fit_service = CulturalFitService()
            culture_matches = fit_service.get_culture_matches(ocean_scores, ctx.deps.derived_metrics, top_n=3)
            top_cultures = [m.culture_type for m in culture_matches]

            # Culture keywords mapping for basic analysis
            culture_keywords = {
                "Startup Disruptor": ["startup", "fast-paced", "disrupt", "agile", "move fast", "growth"],
                "Tech Innovator": ["innovation", "cutting-edge", "technology", "r&d", "research", "ai", "ml"],
                "Corporate Enterprise": ["enterprise", "fortune 500", "established", "global", "corporate"],
                "Creative Agency": ["creative", "design", "brand", "agency", "portfolio", "artistic"],
                "Mission-Driven": ["mission", "impact", "social", "nonprofit", "purpose", "sustainability"],
                "Consulting Firm": ["consulting", "client", "advisory", "strategy", "professional services"],
                "Remote-First": ["remote", "distributed", "work from home", "flexible", "async"],
                "Family Business": ["family", "close-knit", "tradition", "values", "community"],
                "Research & Academic": ["research", "academic", "university", "phd", "science", "publish"],
                "Healthcare": ["healthcare", "medical", "patient", "hospital", "clinical", "health"],
                "Government": ["government", "public sector", "federal", "agency", "civil service"],
                "Entrepreneurial": ["entrepreneurial", "founder", "ownership", "equity", "build"]
            }

            # Build job results with culture fit scores
            job_results = []
            for job in jobs:
                desc_lower = (job.description or "").lower()

                # Determine culture type based on keywords
                best_culture = "Tech Innovator"  # default
                best_score = 0
                for culture, keywords in culture_keywords.items():
                    matches = sum(1 for kw in keywords if kw in desc_lower)
                    if matches > best_score:
                        best_score = matches
                        best_culture = culture

                # Calculate base fit score based on whether it matches user's top cultures
                if best_culture in top_cultures:
                    fit_score = 75 + (top_cultures.index(best_culture) == 0) * 10  # 85 for #1, 75 for #2-3
                else:
                    fit_score = 55 + best_score * 5  # 55-70 range

                # Enhanced scoring with CV data
                skills_matched = 0
                experience_bonus = 0

                if career_profile:
                    # Skills matching bonus
                    user_skills = career_profile.get("key_skills", [])
                    if user_skills:
                        user_skills_lower = set(s.lower() for s in user_skills)
                        job_text = desc_lower + " " + " ".join((job.qualifications or [])).lower()
                        matching_skills = sum(1 for skill in user_skills_lower if skill in job_text)
                        skills_matched = matching_skills
                        fit_score += min(matching_skills * 3, 15)  # Max +15 points for skills

                    # Experience level matching bonus
                    years_exp = career_profile.get("years_experience")
                    if years_exp is not None:
                        # Check seniority alignment
                        if years_exp >= 7 and any(w in desc_lower for w in ["senior", "lead", "principal", "staff", "director", "manager"]):
                            experience_bonus = 5
                        elif 3 <= years_exp < 7 and any(w in desc_lower for w in ["mid", "intermediate", "experienced"]):
                            experience_bonus = 5
                        elif years_exp < 3 and any(w in desc_lower for w in ["junior", "entry", "graduate", "associate", "trainee"]):
                            experience_bonus = 5
                        fit_score += experience_bonus

                # Cap at 98
                fit_score = min(fit_score, 98)

                # Build why_fits message
                why_fits_parts = []
                if best_culture in top_cultures:
                    why_fits_parts.append(f"Matches your {best_culture} culture preference")
                else:
                    why_fits_parts.append(f"Environment: {best_culture}")
                if skills_matched > 0:
                    why_fits_parts.append(f"{skills_matched} skills match")
                if experience_bonus > 0:
                    why_fits_parts.append("experience level fits")

                job_results.append({
                    "title": job.title,
                    "company": job.company_name,
                    "location": job.location,
                    "culture_fit": fit_score,
                    "culture_type": best_culture,
                    "why_fits": " | ".join(why_fits_parts),
                    "skills_matched": skills_matched,
                    "salary": job.salary or "",
                    "posted": job.posted_at or "",
                    "apply_link": job.apply_link or "",
                    "description_snippet": (job.description or "")[:150] + "..."
                })

            # Sort by fit score
            job_results.sort(key=lambda x: x["culture_fit"], reverse=True)

            # Store job results for backend to append as JSON
            jobs_data = job_results[:8]
            store_job_search_results({
                "type": "job_results",
                "jobs": jobs_data,
                "search_role": role,
                "search_location": location
            })

            # Build a summary for the AI to use in its response
            top_jobs = jobs_data[:3]
            job_summaries = []
            for job in top_jobs:
                summary = f"- **{job['title']}** at {job['company']} ({job['culture_fit']}% fit, {job['culture_type']})"
                job_summaries.append(summary)

            return f"""Found {len(jobs_data)} jobs for "{role}" in "{location}".

TOP MATCHES (job cards will appear in sidebar):
{chr(10).join(job_summaries)}

The user's top culture matches are: {', '.join(top_cultures[:2])}.
Write a brief, friendly response mentioning these top jobs and why they fit the user's personality.
Reference specific job titles and companies so you can discuss them in follow-up questions."""

        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            return f"Error searching for jobs: {str(e)}. Please try again."

    return agent


# =============================================================================
# Service Class
# =============================================================================

class PersonalityCoachService:
    """
    Service for managing personality coach conversations.

    Uses hybrid approach for job search:
    - AI writes natural text response (stays in conversation history)
    - Backend appends JSON data for frontend to render job cards
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
        message_history: Optional[List[Dict[str, str]]] = None,
        career_profile: Optional[Dict[str, Any]] = None
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
            career_profile: Optional career profile from uploaded CV

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
            message_history=message_history,
            career_profile=career_profile
        )

        # Prepend CV context to message if available
        cv_context = format_cv_context(career_profile)
        enhanced_message = cv_context + message

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

                result = await self.agent.run(enhanced_message, deps=context, message_history=messages)
            else:
                result = await self.agent.run(enhanced_message, deps=context)

            logger.info(f"Personality coach responded to: {message[:50]}...")

            output = result.output

            # Check if there are job search results to append
            job_results = get_last_job_search_results()
            if job_results:
                import json
                # Append job JSON to the AI's text response for frontend to parse
                json_block = json.dumps(job_results, indent=2)
                output = f"{output}\n\n```json\n{json_block}\n```"

            return output

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
        message_history: Optional[List[Dict[str, str]]] = None,
        career_profile: Optional[Dict[str, Any]] = None
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
            career_profile: Optional career profile from uploaded CV

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
            message_history=message_history,
            career_profile=career_profile
        )

        # Prepend CV context to message if available
        cv_context = format_cv_context(career_profile)
        enhanced_message = cv_context + message

        if cv_context:
            logger.info(f"CV context prepended: {cv_context.strip()}")
            logger.info(f"Enhanced message for agent: {enhanced_message[:300]}...")

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
                enhanced_message,
                deps=context,
                message_history=messages
            ) as result:
                # Stream text response
                async for chunk in result.stream_text(delta=True):
                    yield chunk

            # After streaming is complete, check for job search results to append
            job_results = get_last_job_search_results()
            logger.info(f"Job results after streaming: {job_results is not None}, jobs count: {len(job_results.get('jobs', [])) if job_results else 0}")
            if job_results:
                import json
                # Yield job JSON as a final chunk for frontend to parse
                json_block = json.dumps(job_results, indent=2)
                logger.info(f"Yielding job JSON block ({len(json_block)} chars)")
                yield f"\n\n```json\n{json_block}\n```"

            logger.info(f"Personality coach streamed response to: {message[:50]}...")

        except Exception as e:
            import traceback
            import sys
            logger.error(f"Error in personality coach stream: {e}")
            # Print full traceback to stderr for visibility in Docker logs
            traceback.print_exc(file=sys.stderr)
            print(f"CHAT ERROR TRACEBACK:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            raise


# Singleton instance
_coach_service: Optional[PersonalityCoachService] = None


def get_personality_coach() -> PersonalityCoachService:
    """Get or create singleton personality coach service."""
    global _coach_service
    if _coach_service is None:
        _coach_service = PersonalityCoachService()
    return _coach_service
