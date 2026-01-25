"""
Derived Metrics Service
Calculates composite personality metrics from OCEAN (Big Five) scores
Based on established psychological research and standard formulas
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Result for a single derived metric"""
    name: str
    score: float  # 0-100 scale
    level: str  # "Low", "Moderate", "High"
    description: str


class DerivedMetricsService:
    """
    Service for calculating derived personality metrics from OCEAN scores

    These composite metrics are based on established psychological research:
    - Trust Signaling: Based on Agreeableness and low Neuroticism
    - Social Openness: Based on Extraversion and Openness
    - Empathy Index: Based on Agreeableness and Openness
    - Conflict Avoidance: Based on Agreeableness and low Extraversion (assertiveness)

    Standard formulas follow research from:
    - Costa & McCrae (1992) NEO-PI-R
    - DeYoung et al. (2007) - Big Five Aspects Scale
    - John & Srivastava (1999) - BFI
    """

    def __init__(self):
        """Initialize the derived metrics service"""
        logger.info("DerivedMetricsService initialized")

    @staticmethod
    def normalize_score(score: float) -> float:
        """
        Normalize raw OCEAN score to 0-1 range

        OCEAN scores can be in different ranges depending on the model:
        - Some models output [-1, 1]
        - Some output [0, 1]

        This normalizes to [0, 1] for consistent calculations
        """
        # If score is in [-1, 1] range, convert to [0, 1]
        if score < 0:
            return (score + 1) / 2
        # If already in [0, 1] range
        return min(max(score, 0), 1)

    @staticmethod
    def to_percentage(score: float) -> float:
        """Convert 0-1 score to percentage (0-100)"""
        return round(score * 100, 0)

    @staticmethod
    def get_level(percentage: float, thresholds: tuple = (40, 70)) -> str:
        """
        Categorize percentage into levels

        Args:
            percentage: Score as percentage (0-100)
            thresholds: (low_threshold, high_threshold)

        Returns:
            Level string: "Low", "Moderate", or "High"
        """
        low_thresh, high_thresh = thresholds
        if percentage < low_thresh:
            return "Low"
        elif percentage < high_thresh:
            return "Moderate"
        else:
            return "High"

    def calculate_trust_signaling(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Trust Signaling score

        Formula: Trust Signaling = 0.5A + 0.3(1-N) + 0.2C

        High Agreeableness indicates trustworthy, cooperative behavior
        Low Neuroticism indicates emotional stability, which builds trust
        Conscientiousness adds reliability and dependability

        Research basis:
        - Trustworthy individuals score high on Agreeableness (Evans & Revelle, 2008)
        - Emotional stability (low N) correlates with perceived trustworthiness
        - Conscientiousness correlates with reliability and dependability
        """
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))

        # Invert neuroticism (low N = high emotional stability = more trust)
        emotional_stability = 1 - neuroticism

        # Weighted combination: 0.5A + 0.3(1-N) + 0.2C
        trust_score = (agreeableness * 0.5) + (emotional_stability * 0.3) + (conscientiousness * 0.2)
        percentage = self.to_percentage(trust_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may come across as guarded or skeptical in new relationships.",
            "Moderate": "You balance openness with appropriate caution in building trust.",
            "High": "Your warmth and stability naturally inspire trust in others."
        }

        return MetricResult(
            name="Trust Signaling",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_social_openness(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Social Openness score

        Formula: Social Openness = (Extraversion * 0.5) + (Openness * 0.35) + (Agreeableness * 0.15)

        Extraversion drives social engagement
        Openness drives receptivity to new social experiences
        Agreeableness facilitates social harmony

        Research basis:
        - DeYoung (2006) meta-trait "Plasticity" = E + O
        - Social engagement correlates strongly with E and O
        """
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))

        # Weighted combination
        social_score = (extraversion * 0.5) + (openness * 0.35) + (agreeableness * 0.15)
        percentage = self.to_percentage(social_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You prefer familiar social settings and may need time to warm up.",
            "Moderate": "You're selectively social, engaging when the context feels right.",
            "High": "You thrive in social settings and actively seek new connections."
        }

        return MetricResult(
            name="Social Openness",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_empathy_index(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Empathic Disposition score

        Formula: Empathic Disposition = 0.45A + 0.35O + 0.2(1-N)

        Agreeableness reflects concern for others
        Openness reflects perspective-taking ability
        Emotional stability allows for regulated empathic responses

        Research basis:
        - Graziano et al. (2007) - A predicts prosocial behavior
        - Openness correlates with cognitive empathy (perspective-taking)
        - High N can lead to empathic distress rather than compassion
        """
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))

        # Emotional regulation for healthy empathy
        emotional_regulation = 1 - neuroticism

        # Weighted combination: 0.45A + 0.35O + 0.2(1-N)
        empathy_score = (agreeableness * 0.45) + (openness * 0.35) + (emotional_regulation * 0.2)
        percentage = self.to_percentage(empathy_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You tend to maintain emotional distance and focus on logic over feelings.",
            "Moderate": "You balance empathy with objectivity in understanding others.",
            "High": "You naturally attune to others' emotions and perspectives."
        }

        return MetricResult(
            name="Empathy Index",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_conflict_avoidance(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Conflict Avoidance score

        Formula: Conflict Avoidance = 0.4A + 0.3(1-E) + 0.3N

        High Agreeableness = preference for harmony
        Low Extraversion = less assertive, avoids confrontation
        High Neuroticism = anxiety about conflict

        Research basis:
        - Graziano et al. (1996) - A correlates with conflict avoidance
        - Assertiveness facet of E negatively correlates with avoidance
        - N contributes to conflict anxiety

        Note: High conflict avoidance isn't always positive - it can indicate
        difficulty with healthy confrontation and boundary-setting.
        """
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))

        # Low extraversion contributes to avoidance
        low_assertiveness = 1 - extraversion

        # Weighted combination: 0.4A + 0.3(1-E) + 0.3N
        avoidance_score = (agreeableness * 0.4) + (low_assertiveness * 0.3) + (neuroticism * 0.3)
        percentage = self.to_percentage(avoidance_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You're comfortable with direct confrontation and asserting boundaries.",
            "Moderate": "You pick your battles wisely, balancing harmony with assertiveness.",
            "High": "You strongly prefer harmony and may avoid necessary confrontations."
        }

        return MetricResult(
            name="Conflict Avoidance",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_harmony_seeking(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Harmony Seeking score (optional metric)

        Formula: Harmony Seeking = 0.6A + 0.4(1-E)

        This metric captures avoidance driven by preference for harmony
        rather than anxiety. People high on this prefer peace and cooperation.

        High Agreeableness = values harmony and cooperation
        Low Extraversion = less confrontational, prefers consensus

        Research basis:
        - Agreeableness is the primary driver of harmony-seeking behavior
        - Low assertiveness (facet of E) contributes to preference for agreement
        """
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        # Low extraversion contributes to harmony preference
        low_assertiveness = 1 - extraversion

        # Weighted combination: 0.6A + 0.4(1-E)
        harmony_score = (agreeableness * 0.6) + (low_assertiveness * 0.4)
        percentage = self.to_percentage(harmony_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You prioritize directness over maintaining harmony in interactions.",
            "Moderate": "You value harmony but can be direct when necessary.",
            "High": "You strongly value peaceful relationships and cooperative interactions."
        }

        return MetricResult(
            name="Harmony Seeking",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_anxiety_avoidance(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Anxiety Avoidance score (optional metric)

        Formula: Anxiety Avoidance = 0.7N + 0.3(1-E)

        This metric captures avoidance driven by anxiety and fear of conflict.
        Different from harmony-seeking, this is about stress and discomfort.

        High Neuroticism = anxiety and stress about confrontation
        Low Extraversion = reluctance to engage in assertive behavior

        Research basis:
        - High N individuals experience more distress in conflict situations
        - Low E individuals may avoid confrontation due to social discomfort
        """
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        # Low extraversion contributes to avoidance
        low_assertiveness = 1 - extraversion

        # Weighted combination: 0.7N + 0.3(1-E)
        anxiety_score = (neuroticism * 0.7) + (low_assertiveness * 0.3)
        percentage = self.to_percentage(anxiety_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You handle confrontation with emotional stability and composure.",
            "Moderate": "You may feel some anxiety about conflict but can manage it.",
            "High": "Confrontation tends to cause significant stress or anxiety for you."
        }

        return MetricResult(
            name="Anxiety Avoidance",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    # ============================================
    # Work DNA & Focus Metrics
    # ============================================

    def calculate_persistence(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Persistence score

        Formula: Persistence = 0.6C + 0.25(1-N) + 0.15A

        High Conscientiousness = disciplined, goal-oriented, perseverant
        Low Neuroticism = emotional resilience, doesn't give up easily
        Agreeableness = cooperative persistence, works well with others

        Research basis:
        - Conscientiousness is the strongest predictor of persistence (Duckworth et al., 2007)
        - Emotional stability helps maintain effort under stress
        - Agreeableness contributes to sustained collaborative efforts
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))

        # Emotional stability (inverse of neuroticism)
        emotional_stability = 1 - neuroticism

        # Weighted combination: 0.6C + 0.25(1-N) + 0.15A
        persistence_score = (conscientiousness * 0.6) + (emotional_stability * 0.25) + (agreeableness * 0.15)
        percentage = self.to_percentage(persistence_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may find it challenging to maintain focus on long-term goals when obstacles arise.",
            "Moderate": "You show reasonable persistence but may benefit from strategies to stay motivated.",
            "High": "You demonstrate strong determination and follow-through on commitments."
        }

        return MetricResult(
            name="Persistence",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_focus_attention(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Focus & Attention score

        Formula: Focus & Attention = 0.5C + 0.35(1-N) + 0.15(1-O)

        High Conscientiousness = organized, disciplined, detail-oriented
        Low Neuroticism = calm mind, less mental distraction from anxiety
        Low Openness = less distracted by new ideas, more focused on task at hand

        Research basis:
        - Conscientiousness correlates with sustained attention (DeYoung, 2014)
        - Anxiety (N) interferes with concentration and working memory
        - High O can lead to mind-wandering (Kaufman, 2011)
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        # Low N and low O contribute to focus
        emotional_stability = 1 - neuroticism
        low_openness = 1 - openness

        # Weighted combination: 0.5C + 0.35(1-N) + 0.15(1-O)
        focus_score = (conscientiousness * 0.5) + (emotional_stability * 0.35) + (low_openness * 0.15)
        percentage = self.to_percentage(focus_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may find sustained concentration challenging; try breaking tasks into smaller chunks.",
            "Moderate": "You can maintain focus reasonably well but may benefit from structured work environments.",
            "High": "You excel at sustained attention and can concentrate deeply on tasks."
        }

        return MetricResult(
            name="Focus & Attention",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_structure_preference(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Structure Preference score

        Formula: Structure Preference = 0.6C + 0.4(1-O)

        High Conscientiousness = prefers organization, planning, and order
        Low Openness = prefers routine, conventional approaches, predictability

        Research basis:
        - C is the primary driver of preference for structure (Costa & McCrae, 1992)
        - Low O indicates preference for familiar routines over novelty
        - Together they indicate someone who thrives in structured environments
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        # Low openness contributes to structure preference
        low_openness = 1 - openness

        # Weighted combination: 0.6C + 0.4(1-O)
        structure_score = (conscientiousness * 0.6) + (low_openness * 0.4)
        percentage = self.to_percentage(structure_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You thrive in flexible, dynamic environments and may feel constrained by rigid structures.",
            "Moderate": "You appreciate some structure but also value flexibility when needed.",
            "High": "You work best with clear processes, schedules, and well-defined expectations."
        }

        return MetricResult(
            name="Structure Preference",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_risk_aversion_work(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Risk Aversion (Work) score

        Formula: Risk Aversion (Work) = 0.45N + 0.35(1-E) + 0.2(1-O)

        High Neuroticism = worry about potential negative outcomes
        Low Extraversion = cautious, less impulsive, prefers safe choices
        Low Openness = prefers tried-and-true methods, avoids experimentation

        Research basis:
        - N correlates with risk perception and avoidance (Lauriola & Levin, 2001)
        - Low E (especially low excitement-seeking) relates to cautiousness
        - Low O indicates preference for conventional, low-risk approaches
        """
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        # Low E and low O contribute to risk aversion
        low_extraversion = 1 - extraversion
        low_openness = 1 - openness

        # Weighted combination: 0.45N + 0.35(1-E) + 0.2(1-O)
        risk_aversion_score = (neuroticism * 0.45) + (low_extraversion * 0.35) + (low_openness * 0.2)
        percentage = self.to_percentage(risk_aversion_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You're comfortable taking calculated risks and may seek out challenging opportunities.",
            "Moderate": "You balance caution with willingness to take reasonable risks when justified.",
            "High": "You prefer safe, proven approaches and may need extra support when facing uncertainty."
        }

        return MetricResult(
            name="Risk Aversion",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_all_work_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all work DNA and focus metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores

        Returns:
            Dictionary with all work metrics and coach recommendation
        """
        # Calculate work metrics
        persistence = self.calculate_persistence(ocean_scores)
        focus = self.calculate_focus_attention(ocean_scores)
        structure = self.calculate_structure_preference(ocean_scores)
        risk_aversion = self.calculate_risk_aversion_work(ocean_scores)

        # Generate work coach recommendation
        coach_recommendation = self.generate_work_coach_recommendation(
            ocean_scores, persistence, focus, structure, risk_aversion
        )

        # Generate work actionable steps
        actionable_steps = self.generate_work_actionable_steps(
            ocean_scores, persistence, focus, structure, risk_aversion
        )

        return {
            "metrics": {
                "persistence": {
                    "score": persistence.score,
                    "level": persistence.level,
                    "description": persistence.description
                },
                "focus_attention": {
                    "score": focus.score,
                    "level": focus.level,
                    "description": focus.description
                },
                "structure_preference": {
                    "score": structure.score,
                    "level": structure.level,
                    "description": structure.description
                },
                "risk_aversion": {
                    "score": risk_aversion.score,
                    "level": risk_aversion.level,
                    "description": risk_aversion.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_work_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        persistence: MetricResult,
        focus: MetricResult,
        structure: MetricResult,
        risk_aversion: MetricResult
    ) -> str:
        """
        Generate personalized work coach recommendation based on work metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if persistence.level == "High":
            strengths.append("exceptional determination and follow-through")
        if focus.level == "High":
            strengths.append("deep concentration abilities")
        if structure.level == "High":
            strengths.append("strong organizational skills")
        if risk_aversion.level == "Low":
            strengths.append("comfort with calculated risks")

        # Analyze areas for growth
        growth_areas = []
        if persistence.level == "Low":
            growth_areas.append("Consider setting smaller milestones to maintain motivation on long-term projects.")
        if focus.level == "Low":
            growth_areas.append("Try time-blocking techniques and minimizing distractions to improve concentration.")
        if structure.level == "Low" and focus.level != "High":
            growth_areas.append("A bit more structure in your workflow might help you channel your creative energy productively.")
        if risk_aversion.level == "High":
            growth_areas.append("Practice making small decisions quickly to build confidence in uncertain situations.")

        # Build recommendation
        if strengths:
            recommendations.append(f"Your {' and '.join(strengths)} are valuable assets in the workplace.")

        if persistence.level in ["High", "Moderate"] and focus.level in ["High", "Moderate"]:
            recommendations.append("You have the discipline to see complex projects through to completion.")

        if structure.level == "High" and risk_aversion.level == "High":
            recommendations.append("While your methodical approach ensures quality, consider embracing controlled experimentation to drive innovation.")

        if growth_areas:
            recommendations.append(growth_areas[0])

        return " ".join(recommendations) if recommendations else "Continue developing your professional skills through practice and self-awareness."

    def generate_work_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        persistence: MetricResult,
        focus: MetricResult,
        structure: MetricResult,
        risk_aversion: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps for work skills
        """
        steps = []

        # Persistence development
        if persistence.level != "High":
            steps.append({"emoji": "🎯", "text": "Set small daily goals"})
            steps.append({"emoji": "📊", "text": "Track progress visually"})

        # Focus development
        if focus.level != "High":
            steps.append({"emoji": "⏰", "text": "Use Pomodoro technique"})
            steps.append({"emoji": "📵", "text": "Create distraction-free zones"})

        # Structure preference
        if structure.level == "Low":
            steps.append({"emoji": "📋", "text": "Try a simple daily checklist"})
            steps.append({"emoji": "🗓️", "text": "Schedule focused work blocks"})
        elif structure.level == "High":
            steps.append({"emoji": "🔄", "text": "Allow flexibility for creativity"})

        # Risk aversion
        if risk_aversion.level == "High":
            steps.append({"emoji": "🧪", "text": "Start small experiments"})
            steps.append({"emoji": "💡", "text": "Reframe failures as learning"})
        elif risk_aversion.level == "Low":
            steps.append({"emoji": "📝", "text": "Document risk assessments"})

        # Always include some positive steps
        if persistence.level == "High":
            steps.append({"emoji": "🏆", "text": "Tackle challenging projects"})

        if focus.level == "High":
            steps.append({"emoji": "🧠", "text": "Take on deep work assignments"})

        # Limit to 6 steps
        return steps[:6]

    def calculate_all_relationship_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all relationship and empathy metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores
                         Keys: openness, conscientiousness, extraversion,
                               agreeableness, neuroticism

        Returns:
            Dictionary with all metrics and coach recommendation
        """
        # Calculate core metrics
        trust = self.calculate_trust_signaling(ocean_scores)
        social = self.calculate_social_openness(ocean_scores)
        empathy = self.calculate_empathy_index(ocean_scores)
        conflict = self.calculate_conflict_avoidance(ocean_scores)

        # Calculate optional metrics (sub-components of conflict avoidance)
        harmony = self.calculate_harmony_seeking(ocean_scores)
        anxiety = self.calculate_anxiety_avoidance(ocean_scores)

        # Generate coach recommendation
        coach_recommendation = self.generate_coach_recommendation(
            ocean_scores, trust, social, empathy, conflict, harmony, anxiety
        )

        # Generate actionable steps
        actionable_steps = self.generate_actionable_steps(
            ocean_scores, trust, social, empathy, conflict, harmony, anxiety
        )

        return {
            "metrics": {
                "trust_signaling": {
                    "score": trust.score,
                    "level": trust.level,
                    "description": trust.description
                },
                "social_openness": {
                    "score": social.score,
                    "level": social.level,
                    "description": social.description
                },
                "empathic_disposition": {
                    "score": empathy.score,
                    "level": empathy.level,
                    "description": empathy.description
                },
                "conflict_avoidance": {
                    "score": conflict.score,
                    "level": conflict.level,
                    "description": conflict.description
                },
                "harmony_seeking": {
                    "score": harmony.score,
                    "level": harmony.level,
                    "description": harmony.description
                },
                "anxiety_avoidance": {
                    "score": anxiety.score,
                    "level": anxiety.level,
                    "description": anxiety.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        trust: MetricResult,
        social: MetricResult,
        empathy: MetricResult,
        conflict: MetricResult,
        harmony: MetricResult,
        anxiety: MetricResult
    ) -> str:
        """
        Generate personalized coach recommendation based on metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if trust.level == "High":
            strengths.append("natural warmth and reliability")
        if social.level == "High":
            strengths.append("social energy and approachability")
        if empathy.level == "High":
            strengths.append("deep emotional attunement")
        if harmony.level == "High" and anxiety.level == "Low":
            strengths.append("genuine preference for cooperation")

        # Analyze areas for growth
        growth_areas = []
        if conflict.level == "High":
            # Distinguish between harmony-driven and anxiety-driven avoidance
            if anxiety.level == "High" and harmony.level != "High":
                growth_areas.append("Your conflict avoidance seems driven by anxiety. Consider practicing assertiveness in low-stakes situations to build confidence.")
            elif harmony.level == "High" and anxiety.level != "High":
                growth_areas.append("Your preference for harmony is admirable, but remember that healthy conflict can strengthen relationships. Practice expressing disagreement respectfully.")
            else:
                growth_areas.append("Explore strategies for navigating disagreements constructively while maintaining your composure.")

        if trust.level == "Low":
            growth_areas.append("Building trust gradually through small acts of vulnerability could help deepen your connections.")
        if social.level == "Low":
            growth_areas.append("Starting with smaller social settings might help you build confidence in social interactions.")
        if empathy.level == "Low":
            growth_areas.append("Practicing active listening and asking open-ended questions can help you connect more deeply with others.")
        if anxiety.level == "High":
            growth_areas.append("Consider stress management techniques to help you feel more comfortable in challenging conversations.")

        # Build recommendation
        if strengths:
            recommendations.append(f"Your {' and '.join(strengths)} are powerful assets in building rapport.")

        if empathy.level in ["High", "Moderate"]:
            recommendations.append("Practice active listening and mirroring subtle positive expressions to deepen your connections.")

        if growth_areas:
            recommendations.append(growth_areas[0])  # Add primary growth area

        return " ".join(recommendations) if recommendations else "Continue developing your interpersonal skills through practice and self-reflection."

    def generate_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        trust: MetricResult,
        social: MetricResult,
        empathy: MetricResult,
        conflict: MetricResult,
        harmony: MetricResult,
        anxiety: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps based on metrics
        """
        steps = []

        # Social development steps
        if social.level != "High":
            steps.append({"emoji": "🤝", "text": "Join social or online communities"})
            steps.append({"emoji": "🌱", "text": "Start conversations with new people"})

        # Empathy development steps
        if empathy.level != "High":
            steps.append({"emoji": "👂", "text": "Practice active listening"})
            steps.append({"emoji": "📚", "text": "Read fiction for perspective"})

        # Communication steps
        steps.append({"emoji": "💬", "text": "Use \"I feel\" statements"})

        # Conflict and boundary steps - tailored based on whether anxiety or harmony driven
        if conflict.level == "High":
            if anxiety.level == "High":
                steps.append({"emoji": "🧘", "text": "Practice calming techniques before difficult conversations"})
                steps.append({"emoji": "📝", "text": "Write out your points before confrontations"})
            if harmony.level == "High":
                steps.append({"emoji": "🛡️", "text": "Set small boundaries daily"})
                steps.append({"emoji": "💪", "text": "Practice assertive communication"})

        # Anxiety management
        if anxiety.level == "High":
            steps.append({"emoji": "🌬️", "text": "Use deep breathing in tense moments"})

        # Trust building
        if trust.level != "High":
            steps.append({"emoji": "❤️", "text": "Practice conversations safely"})
            steps.append({"emoji": "🔓", "text": "Share small vulnerabilities"})

        # Always include some positive steps
        if social.level == "High":
            steps.append({"emoji": "🌟", "text": "Mentor others in social skills"})

        if harmony.level == "High" and conflict.level != "High":
            steps.append({"emoji": "🤗", "text": "Use your peacemaking skills to help others"})

        # Limit to 6 steps
        return steps[:6]

    # ============================================
    # Creativity Pulse Metrics
    # ============================================

    def calculate_ideation_power(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Ideation Power score

        Formula: Ideation Power = 0.65O + 0.25E + 0.1(1-C)

        High Openness = imaginative, creative thinking, idea generation
        High Extraversion = confidence to share ideas, brainstorming energy
        Low Conscientiousness = less constrained by rules, more divergent thinking

        Research basis:
        - Openness is the strongest predictor of creativity (Feist, 1998)
        - Extraversion contributes to idea sharing and collaborative creativity
        - Lower C can allow more unconventional thinking (but too low impairs execution)
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))

        # Low conscientiousness contributes slightly to divergent thinking
        low_conscientiousness = 1 - conscientiousness

        # Weighted combination: 0.65O + 0.25E + 0.1(1-C)
        ideation_score = (openness * 0.65) + (extraversion * 0.25) + (low_conscientiousness * 0.1)
        percentage = self.to_percentage(ideation_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may prefer working with established ideas rather than generating new ones.",
            "Moderate": "You have a balanced approach to ideation and can generate creative solutions when needed.",
            "High": "Your high capacity for ideation is a great strength! You naturally generate innovative ideas."
        }

        return MetricResult(
            name="Ideation Power",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_openness_to_novelty(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Openness to Novelty score

        Formula: Openness to Novelty = 0.85O + 0.15E

        High Openness = curiosity, willingness to try new things
        High Extraversion = enthusiasm for new experiences and adventures

        Research basis:
        - Openness is the primary driver of novelty-seeking (McCrae, 1987)
        - Extraversion adds enthusiasm and approach motivation
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        # Weighted combination: 0.85O + 0.15E
        novelty_score = (openness * 0.85) + (extraversion * 0.15)
        percentage = self.to_percentage(novelty_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You prefer familiar approaches and may need encouragement to try new methods.",
            "Moderate": "You balance appreciation for the familiar with openness to new experiences.",
            "High": "You actively seek out new experiences and embrace novel approaches with enthusiasm."
        }

        return MetricResult(
            name="Openness to Novelty",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_originality_index(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Originality Index score

        Formula: Originality Index = 0.55O + 0.25E + 0.2(1-C)

        High Openness = unconventional thinking, unique perspectives
        High Extraversion = confidence to express unique ideas
        Low Conscientiousness = less bound by conventions

        Research basis:
        - Original thinkers score high on O and moderate-low on C (Batey & Furnham, 2006)
        - E contributes to willingness to share original ideas

        Updated formula uses (1-C) instead of -C to ensure full [0,1] output range
        and better differentiation between individuals.
        """
        raw_openness = ocean_scores.get('openness', 0.5)
        raw_extraversion = ocean_scores.get('extraversion', 0.5)
        raw_conscientiousness = ocean_scores.get('conscientiousness', 0.5)

        openness = self.normalize_score(raw_openness)
        extraversion = self.normalize_score(raw_extraversion)
        conscientiousness = self.normalize_score(raw_conscientiousness)

        logger.debug(f"Originality Index inputs - raw O: {raw_openness}, raw E: {raw_extraversion}, raw C: {raw_conscientiousness}")
        logger.debug(f"Originality Index normalized - O: {openness}, E: {extraversion}, C: {conscientiousness}")

        # Low conscientiousness contributes to originality (less bound by conventions)
        low_conscientiousness = 1 - conscientiousness

        # Weighted combination: 0.55O + 0.25E + 0.2(1-C)
        # This formula uses all [0,1] terms and sums to 1.0, ensuring full range output
        originality_score = (openness * 0.55) + (extraversion * 0.25) + (low_conscientiousness * 0.2)
        originality_score = max(0, min(1, originality_score))
        percentage = self.to_percentage(originality_score)

        logger.debug(f"Originality Index result - score: {originality_score}, percentage: {percentage}")
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may prefer conventional solutions and established approaches.",
            "Moderate": "You balance originality with practicality in your creative work.",
            "High": "You naturally produce unique and unconventional ideas that stand out."
        }

        return MetricResult(
            name="Originality Index",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_attention_to_detail_creative(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Attention to Detail (Creative) score

        Formula: Attention to Detail (Creative) = 0.7C + 0.3(1-O)

        High Conscientiousness = detail-oriented, thorough, precise
        Low Openness = focused on specifics rather than big picture

        Research basis:
        - C is the primary driver of attention to detail (Costa & McCrae, 1992)
        - Lower O can increase focus on concrete details vs abstract ideas
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        # Low openness contributes to detail focus
        low_openness = 1 - openness

        # Weighted combination: 0.7C + 0.3(1-O)
        detail_score = (conscientiousness * 0.7) + (low_openness * 0.3)
        percentage = self.to_percentage(detail_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You thrive in the initial ideation phase but may benefit from collaboration for refinement.",
            "Moderate": "You balance big-picture thinking with attention to important details.",
            "High": "You excel at refining and perfecting creative work with meticulous attention to detail."
        }

        return MetricResult(
            name="Attention to Detail (Creative)",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_all_creativity_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all creativity pulse metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores

        Returns:
            Dictionary with all creativity metrics and coach recommendation
        """
        # Calculate creativity metrics
        ideation = self.calculate_ideation_power(ocean_scores)
        novelty = self.calculate_openness_to_novelty(ocean_scores)
        originality = self.calculate_originality_index(ocean_scores)
        detail = self.calculate_attention_to_detail_creative(ocean_scores)

        # Generate creativity coach recommendation
        coach_recommendation = self.generate_creativity_coach_recommendation(
            ocean_scores, ideation, novelty, originality, detail
        )

        # Generate creativity actionable steps
        actionable_steps = self.generate_creativity_actionable_steps(
            ocean_scores, ideation, novelty, originality, detail
        )

        return {
            "metrics": {
                "ideation_power": {
                    "score": ideation.score,
                    "level": ideation.level,
                    "description": ideation.description
                },
                "openness_to_novelty": {
                    "score": novelty.score,
                    "level": novelty.level,
                    "description": novelty.description
                },
                "originality_index": {
                    "score": originality.score,
                    "level": originality.level,
                    "description": originality.description
                },
                "attention_to_detail_creative": {
                    "score": detail.score,
                    "level": detail.level,
                    "description": detail.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_creativity_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        ideation: MetricResult,
        novelty: MetricResult,
        originality: MetricResult,
        detail: MetricResult
    ) -> str:
        """
        Generate personalized creativity coach recommendation based on creativity metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if ideation.level == "High":
            strengths.append("exceptional capacity for ideation")
        if novelty.level == "High":
            strengths.append("enthusiasm for new experiences")
        if originality.level == "High":
            strengths.append("unique and unconventional thinking")
        if detail.level == "High":
            strengths.append("meticulous attention to refinement")

        # Analyze areas for growth
        growth_areas = []
        if ideation.level == "Low":
            growth_areas.append("Try brainstorming techniques or mind mapping to unlock your creative potential.")
        if novelty.level == "Low":
            growth_areas.append("Experiment with small changes to your routine to build comfort with novelty.")
        if originality.level == "Low":
            growth_areas.append("Challenge yourself to find unconventional solutions before settling on the obvious choice.")
        if detail.level == "Low":
            growth_areas.append("Be mindful that a lower score in attention to detail might mean you thrive in the initial ideation phase but may benefit from collaboration for refinement.")

        # Build recommendation
        if strengths:
            recommendations.append(f"Your {' and '.join(strengths)} are great creative assets!")

        if ideation.level == "High" and detail.level == "Low":
            recommendations.append("Try brainstorming techniques or mind mapping to capture and develop your unique ideas.")
        elif ideation.level == "High" and detail.level == "High":
            recommendations.append("You have a rare combination of idea generation and refinement abilities.")

        if originality.level == "High" and novelty.level == "High":
            recommendations.append("Your openness to new experiences fuels your original thinking.")

        if growth_areas:
            recommendations.append(growth_areas[0])

        return " ".join(recommendations) if recommendations else "Continue exploring your creative potential through experimentation and practice."

    def generate_creativity_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        ideation: MetricResult,
        novelty: MetricResult,
        originality: MetricResult,
        detail: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps for creativity
        """
        steps = []

        # Ideation development
        if ideation.level != "High":
            steps.append({"emoji": "🧠", "text": "Practice daily brainstorming"})
            steps.append({"emoji": "🗺️", "text": "Use mind mapping techniques"})
        else:
            steps.append({"emoji": "💡", "text": "Capture ideas immediately"})

        # Novelty seeking
        if novelty.level != "High":
            steps.append({"emoji": "🌱", "text": "Try one new thing weekly"})
            steps.append({"emoji": "🎨", "text": "Explore unfamiliar creative mediums"})

        # Originality development
        if originality.level != "High":
            steps.append({"emoji": "🔄", "text": "Challenge conventional thinking"})
            steps.append({"emoji": "🎲", "text": "Use random prompts for ideas"})
        else:
            steps.append({"emoji": "🌟", "text": "Share your unique perspectives"})

        # Detail orientation
        if detail.level == "Low":
            steps.append({"emoji": "🤝", "text": "Partner with detail-oriented people"})
            steps.append({"emoji": "📝", "text": "Create refinement checklists"})
        elif detail.level == "High":
            steps.append({"emoji": "⏱️", "text": "Set ideation time limits"})

        # Always include some positive steps
        if ideation.level == "High":
            steps.append({"emoji": "🚀", "text": "Lead creative brainstorming sessions"})

        if novelty.level == "High":
            steps.append({"emoji": "🔬", "text": "Experiment with new approaches"})

        # Limit to 6 steps
        return steps[:6]

    # ============================================
    # Stress & Resilience Metrics
    # ============================================

    def calculate_stress_indicators(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Stress Indicators score

        Formula: Stress Indicators = 0.7N + 0.3(1-C)

        High Neuroticism = prone to stress, anxiety, and negative emotions
        Low Conscientiousness = less organized, may feel overwhelmed by demands

        Research basis:
        - Neuroticism is the strongest predictor of stress reactivity (Bolger & Zuckerman, 1995)
        - Low C can contribute to feeling disorganized and stressed
        """
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))

        # Low conscientiousness contributes to stress
        low_conscientiousness = 1 - conscientiousness

        # Weighted combination: 0.7N + 0.3(1-C)
        stress_score = (neuroticism * 0.7) + (low_conscientiousness * 0.3)
        percentage = self.to_percentage(stress_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You tend to remain calm under pressure and handle stressors effectively.",
            "Moderate": "You experience typical levels of stress and generally cope well.",
            "High": "You may be more sensitive to stress and could benefit from stress management techniques."
        }

        return MetricResult(
            name="Stress Indicators",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_emotional_regulation(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Emotional Regulation score

        Formula: Emotional Regulation = 0.45(1-N) + 0.35C + 0.2A

        Low Neuroticism = emotional stability, better mood regulation
        High Conscientiousness = self-discipline, impulse control
        High Agreeableness = social harmony, less reactive in conflicts

        Research basis:
        - Emotional stability (low N) is key for regulation (Gross & John, 2003)
        - C reflects self-control and ability to manage impulses
        - A contributes to smoother interpersonal emotional exchanges
        """
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        agreeableness = self.normalize_score(ocean_scores.get('agreeableness', 0.5))

        # Low neuroticism = emotional stability
        emotional_stability = 1 - neuroticism

        # Weighted combination: 0.45(1-N) + 0.35C + 0.2A
        regulation_score = (emotional_stability * 0.45) + (conscientiousness * 0.35) + (agreeableness * 0.2)
        percentage = self.to_percentage(regulation_score)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "You may find it challenging to manage intense emotions; mindfulness practices could help.",
            "Moderate": "You have a reasonable ability to regulate emotions but may struggle during high stress.",
            "High": "You excel at managing your emotional responses and maintaining composure."
        }

        return MetricResult(
            name="Emotional Regulation",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_resilience_score(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Resilience Score

        Formula: Resilience Score = 0.5C + 0.3(1-N) + 0.2O

        High Conscientiousness = perseverance, goal-directed behavior
        Low Neuroticism = emotional stability, bounces back from setbacks
        High Openness = adaptability, finding new solutions

        Research basis:
        - Resilience correlates with C (discipline) and low N (stability) (Oshio et al., 2018)
        - O contributes to cognitive flexibility and adaptation
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        # Low neuroticism = emotional stability
        emotional_stability = 1 - neuroticism

        # Weighted combination: 0.5C + 0.3(1-N) + 0.2O
        resilience = (conscientiousness * 0.5) + (emotional_stability * 0.3) + (openness * 0.2)
        percentage = self.to_percentage(resilience)
        level = self.get_level(percentage)

        descriptions = {
            "Low": "Building coping strategies and support networks could help you bounce back from challenges.",
            "Moderate": "You have reasonable resilience but may benefit from strengthening your coping toolkit.",
            "High": "You demonstrate strong resilience and recover well from setbacks and challenges."
        }

        return MetricResult(
            name="Resilience Score",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_all_stress_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all stress and resilience metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores

        Returns:
            Dictionary with all stress metrics and coach recommendation
        """
        # Calculate stress metrics
        stress = self.calculate_stress_indicators(ocean_scores)
        regulation = self.calculate_emotional_regulation(ocean_scores)
        resilience = self.calculate_resilience_score(ocean_scores)

        # Generate stress coach recommendation
        coach_recommendation = self.generate_stress_coach_recommendation(
            ocean_scores, stress, regulation, resilience
        )

        # Generate stress actionable steps
        actionable_steps = self.generate_stress_actionable_steps(
            ocean_scores, stress, regulation, resilience
        )

        return {
            "metrics": {
                "stress_indicators": {
                    "score": stress.score,
                    "level": stress.level,
                    "description": stress.description
                },
                "emotional_regulation": {
                    "score": regulation.score,
                    "level": regulation.level,
                    "description": regulation.description
                },
                "resilience_score": {
                    "score": resilience.score,
                    "level": resilience.level,
                    "description": resilience.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_stress_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        stress: MetricResult,
        regulation: MetricResult,
        resilience: MetricResult
    ) -> str:
        """
        Generate personalized stress coach recommendation based on stress metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if stress.level == "Low":
            strengths.append("natural calmness under pressure")
        if regulation.level == "High":
            strengths.append("excellent emotional self-control")
        if resilience.level == "High":
            strengths.append("strong ability to bounce back from challenges")

        # Analyze areas for growth
        growth_areas = []
        if stress.level == "High":
            growth_areas.append("Consider incorporating daily stress-reduction practices like meditation or exercise.")
        if regulation.level == "Low":
            growth_areas.append("Mindfulness and breathing exercises can help you manage intense emotions more effectively.")
        if resilience.level == "Low":
            growth_areas.append("Building a support network and developing coping strategies can strengthen your resilience.")

        # Build recommendation
        if strengths:
            recommendations.append(f"Your {' and '.join(strengths)} are valuable assets for handling life's challenges.")

        if stress.level == "High" and regulation.level == "High":
            recommendations.append("While you may experience stress, your strong emotional regulation helps you manage it effectively.")

        if resilience.level == "High" and stress.level != "Low":
            recommendations.append("Your resilience helps you recover from stressful periods successfully.")

        if regulation.level == "Low" and stress.level == "High":
            recommendations.append("Developing emotional regulation skills could significantly reduce your stress experience.")

        if growth_areas:
            recommendations.append(growth_areas[0])

        return " ".join(recommendations) if recommendations else "Continue developing your stress management skills through practice and self-awareness."

    def generate_stress_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        stress: MetricResult,
        regulation: MetricResult,
        resilience: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps for stress management
        """
        steps = []

        # Stress management
        if stress.level == "High":
            steps.append({"emoji": "🧘", "text": "Practice daily meditation"})
            steps.append({"emoji": "🏃", "text": "Exercise regularly"})
            steps.append({"emoji": "😴", "text": "Prioritize quality sleep"})
        elif stress.level == "Moderate":
            steps.append({"emoji": "🌬️", "text": "Use breathing techniques"})
            steps.append({"emoji": "📅", "text": "Schedule relaxation time"})

        # Emotional regulation
        if regulation.level != "High":
            steps.append({"emoji": "📝", "text": "Keep an emotion journal"})
            steps.append({"emoji": "⏸️", "text": "Pause before reacting"})
        else:
            steps.append({"emoji": "🤝", "text": "Help others regulate emotions"})

        # Resilience building
        if resilience.level != "High":
            steps.append({"emoji": "💪", "text": "Set small achievable goals"})
            steps.append({"emoji": "🌐", "text": "Build a support network"})
            steps.append({"emoji": "🔄", "text": "Reframe setbacks as learning"})
        else:
            steps.append({"emoji": "🏆", "text": "Take on challenging projects"})

        # Always include positive coping steps
        if stress.level == "Low":
            steps.append({"emoji": "🌟", "text": "Share your calm with others"})

        if resilience.level == "High":
            steps.append({"emoji": "🎯", "text": "Mentor others through challenges"})

        # Limit to 6 steps
        return steps[:6]

    # ==========================================
    # OPENNESS TO EXPERIENCE METRICS
    # ==========================================

    def calculate_openness_to_experience(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Openness to Experience score (primary metric)

        Formula: Openness to Experience = O (pure Openness score)

        This is the direct measure of intellectual curiosity, creativity,
        and preference for novelty and variety.
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        score = openness
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You have exceptional curiosity and openness to new experiences, ideas, and perspectives.",
            "Moderate": "You balance curiosity with practicality, open to new ideas while valuing familiar approaches.",
            "Low": "You prefer familiar, proven approaches and value tradition and stability."
        }

        return MetricResult(
            name="openness_to_experience",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_novelty_seeking(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Novelty Seeking score

        Formula: Novelty Seeking = 0.65O + 0.35E

        Combines Openness (curiosity) with Extraversion (adventure-seeking).
        High O drives intellectual novelty; high E drives experiential novelty.
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        score = 0.65 * openness + 0.35 * extraversion
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You actively seek out new experiences, adventures, and unexplored territories.",
            "Moderate": "You enjoy novelty in moderation, balancing exploration with comfort.",
            "Low": "You prefer familiar environments and established routines over new experiences."
        }

        return MetricResult(
            name="novelty_seeking",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_risk_tolerance_adventure(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Risk Tolerance (Adventure) score

        Formula: Risk Tolerance = 0.45O + 0.35E + 0.2(1-N)

        Combines Openness (willingness to try new things), Extraversion (boldness),
        and low Neuroticism (emotional stability under uncertainty).
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))

        score = 0.45 * openness + 0.35 * extraversion + 0.2 * (1 - neuroticism)
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You embrace uncertainty and are comfortable taking calculated risks for growth.",
            "Moderate": "You take measured risks when the potential reward is clear.",
            "Low": "You prefer predictable outcomes and carefully avoid unnecessary risks."
        }

        return MetricResult(
            name="risk_tolerance_adventure",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_planning_preference(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Planning Preference score

        Formula: Planning Preference = 0.6C + 0.4(1-O)

        High Conscientiousness indicates preference for structure and planning.
        Low Openness indicates preference for established methods over improvisation.
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        score = 0.6 * conscientiousness + 0.4 * (1 - openness)
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You strongly prefer detailed planning, structure, and organized approaches.",
            "Moderate": "You balance planning with flexibility, adapting as situations evolve.",
            "Low": "You prefer spontaneity and flexibility over rigid plans and schedules."
        }

        return MetricResult(
            name="planning_preference",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_all_openness_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all openness to experience metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores

        Returns:
            Dictionary with all openness metrics and coach recommendation
        """
        # Calculate openness metrics
        openness_exp = self.calculate_openness_to_experience(ocean_scores)
        novelty = self.calculate_novelty_seeking(ocean_scores)
        risk = self.calculate_risk_tolerance_adventure(ocean_scores)
        planning = self.calculate_planning_preference(ocean_scores)

        # Generate openness coach recommendation
        coach_recommendation = self.generate_openness_coach_recommendation(
            ocean_scores, openness_exp, novelty, risk, planning
        )

        # Generate openness actionable steps
        actionable_steps = self.generate_openness_actionable_steps(
            ocean_scores, openness_exp, novelty, risk, planning
        )

        return {
            "metrics": {
                "openness_to_experience": {
                    "score": openness_exp.score,
                    "level": openness_exp.level,
                    "description": openness_exp.description
                },
                "novelty_seeking": {
                    "score": novelty.score,
                    "level": novelty.level,
                    "description": novelty.description
                },
                "risk_tolerance_adventure": {
                    "score": risk.score,
                    "level": risk.level,
                    "description": risk.description
                },
                "planning_preference": {
                    "score": planning.score,
                    "level": planning.level,
                    "description": planning.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_openness_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        openness_exp: MetricResult,
        novelty: MetricResult,
        risk: MetricResult,
        planning: MetricResult
    ) -> str:
        """
        Generate personalized openness coach recommendation based on openness metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if openness_exp.level == "High":
            strengths.append("exceptional curiosity and openness to ideas")
        if novelty.level == "High":
            strengths.append("enthusiasm for new experiences")
        if risk.level == "High":
            strengths.append("comfort with uncertainty and adventure")
        if planning.level == "High":
            strengths.append("strong organizational and planning skills")

        # Analyze areas for growth
        growth_areas = []
        if openness_exp.level == "Low":
            growth_areas.append("Try exploring one new idea or perspective each week to gradually expand your horizons.")
        if novelty.level == "Low":
            growth_areas.append("Small experiments with new activities can build comfort with novelty over time.")
        if risk.level == "Low":
            growth_areas.append("Start with low-stakes experiments to build confidence in handling uncertainty.")
        if planning.level == "Low":
            growth_areas.append("Brief planning sessions before activities can help balance spontaneity with structure.")

        # Build recommendation
        if strengths:
            recommendations.append(f"Your {' and '.join(strengths[:2])} are valuable assets for growth and exploration!")

        if openness_exp.level == "High" and planning.level == "Low":
            recommendations.append("Your creativity flows best with flexibility. Consider light frameworks to capture your best ideas.")
        elif openness_exp.level == "High" and planning.level == "High":
            recommendations.append("You have a rare balance of curiosity and organization—use it to systematically explore new territories.")

        if novelty.level == "High" and risk.level == "High":
            recommendations.append("Your adventurous spirit opens doors to unique experiences and growth opportunities.")

        if growth_areas:
            recommendations.append(growth_areas[0])

        return " ".join(recommendations) if recommendations else "Continue exploring new ideas and experiences at your own comfortable pace."

    def generate_openness_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        openness_exp: MetricResult,
        novelty: MetricResult,
        risk: MetricResult,
        planning: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps for openness to experience
        """
        steps = []

        # Openness development
        if openness_exp.level != "High":
            steps.append({"emoji": "📚", "text": "Read outside your usual genres"})
            steps.append({"emoji": "🎨", "text": "Try a new creative hobby"})
        else:
            steps.append({"emoji": "💡", "text": "Share your insights with others"})
            steps.append({"emoji": "🌍", "text": "Explore different cultural perspectives"})

        # Novelty seeking
        if novelty.level != "High":
            steps.append({"emoji": "🗺️", "text": "Visit a new place monthly"})
            steps.append({"emoji": "🍽️", "text": "Try unfamiliar cuisines"})
        else:
            steps.append({"emoji": "🚀", "text": "Pursue ambitious new projects"})

        # Risk tolerance
        if risk.level == "Low":
            steps.append({"emoji": "🎲", "text": "Take small calculated risks"})
            steps.append({"emoji": "🌱", "text": "Embrace learning from failures"})
        elif risk.level == "High":
            steps.append({"emoji": "⚖️", "text": "Balance adventure with reflection"})

        # Planning balance
        if planning.level == "Low":
            steps.append({"emoji": "📝", "text": "Try light planning frameworks"})
        elif planning.level == "High":
            steps.append({"emoji": "🎭", "text": "Schedule unplanned exploration time"})

        # Always include positive growth steps
        if openness_exp.level == "High":
            steps.append({"emoji": "🎯", "text": "Mentor others in creative thinking"})

        if novelty.level == "High" and risk.level == "High":
            steps.append({"emoji": "✈️", "text": "Plan an adventure trip"})

        # Limit to 6 steps
        return steps[:6]

    # ==========================================
    # LEARNING & GROWTH METRICS
    # ==========================================

    def calculate_intellectual_curiosity(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Intellectual Curiosity score

        Formula: Intellectual Curiosity = 0.8O + 0.2E

        High Openness drives intellectual exploration and curiosity.
        Extraversion adds engagement and discussion of ideas with others.
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        score = 0.8 * openness + 0.2 * extraversion
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You have a strong drive to learn, explore ideas, and understand how things work.",
            "Moderate": "You enjoy learning when topics interest you, balancing curiosity with practicality.",
            "Low": "You prefer practical, applicable knowledge over abstract exploration."
        }

        return MetricResult(
            name="intellectual_curiosity",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_reflective_tendency(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Reflective Tendency score

        Formula: Reflective Tendency = 0.45O + 0.35C + 0.2(1-E)

        High Openness enables deep thinking and introspection.
        High Conscientiousness adds analytical rigor.
        Low Extraversion (introversion) provides space for reflection.
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        score = 0.45 * openness + 0.35 * conscientiousness + 0.2 * (1 - extraversion)
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You naturally engage in deep reflection and thoughtful analysis of experiences.",
            "Moderate": "You reflect on experiences when prompted, balancing action with contemplation.",
            "Low": "You prefer action over reflection, learning through doing rather than analyzing."
        }

        return MetricResult(
            name="reflective_tendency",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_structured_learning_preference(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Structured Learning Preference score

        Formula: Structured Learning Preference = 0.65C + 0.35(1-O)

        High Conscientiousness indicates preference for organized, systematic learning.
        Low Openness indicates preference for proven methods and structured curricula.
        """
        conscientiousness = self.normalize_score(ocean_scores.get('conscientiousness', 0.5))
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))

        score = 0.65 * conscientiousness + 0.35 * (1 - openness)
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You thrive with clear curricula, defined objectives, and systematic progression.",
            "Moderate": "You appreciate structure but can adapt to different learning formats.",
            "Low": "You prefer flexible, self-directed learning over rigid structures."
        }

        return MetricResult(
            name="structured_learning_preference",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_adaptability_index(self, ocean_scores: Dict[str, float]) -> MetricResult:
        """
        Calculate Adaptability Index score

        Formula: Adaptability Index = 0.45O + 0.35(1-N) + 0.2E

        High Openness enables flexibility and acceptance of new approaches.
        Low Neuroticism provides emotional stability during change.
        Extraversion adds resilience through social support and engagement.
        """
        openness = self.normalize_score(ocean_scores.get('openness', 0.5))
        neuroticism = self.normalize_score(ocean_scores.get('neuroticism', 0.5))
        extraversion = self.normalize_score(ocean_scores.get('extraversion', 0.5))

        score = 0.45 * openness + 0.35 * (1 - neuroticism) + 0.2 * extraversion
        percentage = self.to_percentage(score)
        level = self.get_level(percentage)

        descriptions = {
            "High": "You adapt quickly to new learning environments and changing requirements.",
            "Moderate": "You can adapt to change with some adjustment time and support.",
            "Low": "You prefer consistency and may need extra time to adjust to new approaches."
        }

        return MetricResult(
            name="adaptability_index",
            score=percentage,
            level=level,
            description=descriptions[level]
        )

    def calculate_all_learning_metrics(
        self,
        ocean_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate all learning and growth metrics

        Args:
            ocean_scores: Dictionary with OCEAN trait scores

        Returns:
            Dictionary with all learning metrics and coach recommendation
        """
        # Calculate learning metrics
        curiosity = self.calculate_intellectual_curiosity(ocean_scores)
        reflective = self.calculate_reflective_tendency(ocean_scores)
        structured = self.calculate_structured_learning_preference(ocean_scores)
        adaptability = self.calculate_adaptability_index(ocean_scores)

        # Generate learning coach recommendation
        coach_recommendation = self.generate_learning_coach_recommendation(
            ocean_scores, curiosity, reflective, structured, adaptability
        )

        # Generate learning actionable steps
        actionable_steps = self.generate_learning_actionable_steps(
            ocean_scores, curiosity, reflective, structured, adaptability
        )

        return {
            "metrics": {
                "intellectual_curiosity": {
                    "score": curiosity.score,
                    "level": curiosity.level,
                    "description": curiosity.description
                },
                "reflective_tendency": {
                    "score": reflective.score,
                    "level": reflective.level,
                    "description": reflective.description
                },
                "structured_learning_preference": {
                    "score": structured.score,
                    "level": structured.level,
                    "description": structured.description
                },
                "adaptability_index": {
                    "score": adaptability.score,
                    "level": adaptability.level,
                    "description": adaptability.description
                }
            },
            "coach_recommendation": coach_recommendation,
            "actionable_steps": actionable_steps
        }

    def generate_learning_coach_recommendation(
        self,
        ocean_scores: Dict[str, float],
        curiosity: MetricResult,
        reflective: MetricResult,
        structured: MetricResult,
        adaptability: MetricResult
    ) -> str:
        """
        Generate personalized learning coach recommendation based on learning metrics
        """
        recommendations = []

        # Analyze strengths
        strengths = []
        if curiosity.level == "High":
            strengths.append("strong intellectual curiosity")
        if reflective.level == "High":
            strengths.append("reflective nature")
        if structured.level == "High":
            strengths.append("appreciation for structured learning")
        if adaptability.level == "High":
            strengths.append("adaptability to new approaches")

        # Analyze areas for growth
        growth_areas = []
        if curiosity.level == "Low":
            growth_areas.append("Try connecting new topics to practical applications to spark interest.")
        if reflective.level == "Low":
            growth_areas.append("Schedule brief reflection time after learning sessions to deepen understanding.")
        if structured.level == "Low":
            growth_areas.append("Light frameworks can help organize your flexible learning style.")
        if adaptability.level == "Low":
            growth_areas.append("Gradual exposure to new methods can build comfort with change.")

        # Build recommendation
        if strengths:
            if len(strengths) >= 2:
                recommendations.append(f"Your {strengths[0]} and {strengths[1]} make you a natural learner.")
            else:
                recommendations.append(f"Your {strengths[0]} is a valuable asset for growth.")

        if curiosity.level == "High" and structured.level == "Low":
            recommendations.append("Embrace diverse sources of information and be open to unconventional learning paths, as your preference might lean away from highly structured environments.")
        elif curiosity.level == "High" and structured.level == "High":
            recommendations.append("You combine curiosity with discipline—ideal for mastering complex subjects systematically.")

        if reflective.level == "High" and adaptability.level == "High":
            recommendations.append("Your reflective nature combined with adaptability helps you learn from any situation.")

        if growth_areas:
            recommendations.append(growth_areas[0])

        return " ".join(recommendations) if recommendations else "Continue exploring learning approaches that match your natural style and interests."

    def generate_learning_actionable_steps(
        self,
        ocean_scores: Dict[str, float],
        curiosity: MetricResult,
        reflective: MetricResult,
        structured: MetricResult,
        adaptability: MetricResult
    ) -> List[Dict[str, str]]:
        """
        Generate actionable development steps for learning and growth
        """
        steps = []

        # Intellectual curiosity development
        if curiosity.level != "High":
            steps.append({"emoji": "📚", "text": "Read outside your usual genres"})
            steps.append({"emoji": "🎧", "text": "Listen to educational podcasts"})
        else:
            steps.append({"emoji": "🔬", "text": "Explore interdisciplinary topics"})
            steps.append({"emoji": "🎓", "text": "Take advanced courses in interests"})

        # Reflective tendency
        if reflective.level != "High":
            steps.append({"emoji": "📝", "text": "Keep a learning journal"})
            steps.append({"emoji": "🤔", "text": "Schedule weekly reflection time"})
        else:
            steps.append({"emoji": "💭", "text": "Share insights with others"})

        # Structured learning
        if structured.level == "Low":
            steps.append({"emoji": "📋", "text": "Try light planning frameworks"})
        elif structured.level == "High":
            steps.append({"emoji": "🎯", "text": "Set clear learning milestones"})

        # Adaptability
        if adaptability.level != "High":
            steps.append({"emoji": "🌱", "text": "Try one new learning method monthly"})
            steps.append({"emoji": "🔄", "text": "Practice learning in different settings"})
        else:
            steps.append({"emoji": "🚀", "text": "Mentor others in adapting to change"})

        # Always include community-based learning
        steps.append({"emoji": "🤝", "text": "Join social or online communities"})
        steps.append({"emoji": "💬", "text": "Start conversations with new people"})

        # Limit to 6 steps
        return steps[:6]


# Singleton instance
_derived_metrics_service: Optional[DerivedMetricsService] = None


def get_derived_metrics_service() -> DerivedMetricsService:
    """Get or create singleton derived metrics service instance"""
    global _derived_metrics_service
    if _derived_metrics_service is None:
        _derived_metrics_service = DerivedMetricsService()
    return _derived_metrics_service
