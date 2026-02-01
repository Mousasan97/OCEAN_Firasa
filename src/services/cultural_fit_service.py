"""
Cultural Fit Service

Matches user personality profiles to workplace culture types based on OCEAN scores
and derived metrics. Uses cosine similarity to find best culture type matches.

Culture Taxonomy:
- 12 culture types mapped across 8 dimensions
- Each dimension computed from weighted OCEAN + derived metrics combinations
- Cosine similarity matching for person-to-culture fit scoring
"""
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from src.utils.logger import get_logger
from src.services.derived_metrics_service import get_derived_metrics_service

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CultureMatch:
    """Result for a single culture type match"""
    culture_type: str
    fit_score: float  # 0-100
    description: str
    strengths: List[str]
    potential_challenges: List[str]
    example_companies: List[str]


@dataclass
class CultureDimensions:
    """The 8 culture dimensions computed from personality"""
    innovation_vs_stability: float      # -1 (stability) to +1 (innovation)
    collaboration_vs_independence: float # -1 (independence) to +1 (collaboration)
    hierarchy_vs_flat: float            # -1 (flat) to +1 (hierarchy)
    fast_paced_vs_steady: float         # -1 (steady) to +1 (fast-paced)
    results_vs_process: float           # -1 (process) to +1 (results)
    formal_vs_casual: float             # -1 (casual) to +1 (formal)
    risk_taking_vs_cautious: float      # -1 (cautious) to +1 (risk-taking)
    work_life_balance_vs_intensity: float  # -1 (intensity) to +1 (balance)


# =============================================================================
# Culture Type Definitions
# =============================================================================

CULTURE_TYPES = {
    "startup_disruptor": {
        "name": "Startup Disruptor",
        "description": "Fast-moving, innovative environments that challenge the status quo. Thrives on ambiguity and rapid change.",
        "vector": {
            "innovation_vs_stability": 0.9,
            "collaboration_vs_independence": 0.3,
            "hierarchy_vs_flat": -0.8,
            "fast_paced_vs_steady": 0.9,
            "results_vs_process": 0.7,
            "formal_vs_casual": -0.8,
            "risk_taking_vs_cautious": 0.9,
            "work_life_balance_vs_intensity": -0.6
        },
        "example_companies": ["Early-stage startups", "Tech disruptors", "Venture-backed companies"],
        "strengths": ["Rapid innovation", "Flexibility", "Direct impact", "Growth opportunities"],
        "challenges": ["Uncertainty", "Long hours", "Limited structure", "Resource constraints"]
    },
    "tech_innovator": {
        "name": "Tech Innovator",
        "description": "Technology-first companies focused on cutting-edge solutions. Values technical excellence and creative problem-solving.",
        "vector": {
            "innovation_vs_stability": 0.8,
            "collaboration_vs_independence": 0.4,
            "hierarchy_vs_flat": -0.5,
            "fast_paced_vs_steady": 0.6,
            "results_vs_process": 0.5,
            "formal_vs_casual": -0.6,
            "risk_taking_vs_cautious": 0.6,
            "work_life_balance_vs_intensity": 0.0
        },
        "example_companies": ["Google", "Meta", "Microsoft", "Apple"],
        "strengths": ["Technical challenge", "Innovation culture", "Resources", "Peer learning"],
        "challenges": ["High expectations", "Competition", "Rapid change", "Scale complexity"]
    },
    "corporate_enterprise": {
        "name": "Corporate Enterprise",
        "description": "Large, established organizations with clear structures and processes. Offers stability and career progression.",
        "vector": {
            "innovation_vs_stability": -0.6,
            "collaboration_vs_independence": 0.3,
            "hierarchy_vs_flat": 0.7,
            "fast_paced_vs_steady": -0.5,
            "results_vs_process": -0.4,
            "formal_vs_casual": 0.7,
            "risk_taking_vs_cautious": -0.7,
            "work_life_balance_vs_intensity": 0.3
        },
        "example_companies": ["Fortune 500 companies", "Banks", "Insurance", "Consulting firms"],
        "strengths": ["Stability", "Clear paths", "Benefits", "Training programs"],
        "challenges": ["Bureaucracy", "Slow change", "Politics", "Limited autonomy"]
    },
    "creative_agency": {
        "name": "Creative Agency",
        "description": "Design and creative-focused environments that value aesthetics, originality, and artistic expression.",
        "vector": {
            "innovation_vs_stability": 0.7,
            "collaboration_vs_independence": 0.5,
            "hierarchy_vs_flat": -0.4,
            "fast_paced_vs_steady": 0.4,
            "results_vs_process": 0.3,
            "formal_vs_casual": -0.7,
            "risk_taking_vs_cautious": 0.5,
            "work_life_balance_vs_intensity": -0.2
        },
        "example_companies": ["Design studios", "Ad agencies", "Media companies", "Branding firms"],
        "strengths": ["Creative freedom", "Portfolio building", "Diverse projects", "Artistic culture"],
        "challenges": ["Client demands", "Deadlines", "Subjectivity", "Income variability"]
    },
    "mission_driven": {
        "name": "Mission-Driven",
        "description": "Purpose-led organizations focused on social impact. Values meaning and contribution over profit.",
        "vector": {
            "innovation_vs_stability": 0.2,
            "collaboration_vs_independence": 0.7,
            "hierarchy_vs_flat": -0.3,
            "fast_paced_vs_steady": 0.0,
            "results_vs_process": 0.2,
            "formal_vs_casual": -0.3,
            "risk_taking_vs_cautious": 0.1,
            "work_life_balance_vs_intensity": 0.4
        },
        "example_companies": ["Non-profits", "Social enterprises", "B-Corps", "NGOs"],
        "strengths": ["Meaningful work", "Community", "Values alignment", "Purpose"],
        "challenges": ["Limited resources", "Slower pay growth", "Emotional demands", "Burnout risk"]
    },
    "consulting_professional": {
        "name": "Consulting/Professional Services",
        "description": "Client-focused professional environments. High standards, diverse projects, and continuous learning.",
        "vector": {
            "innovation_vs_stability": 0.1,
            "collaboration_vs_independence": 0.4,
            "hierarchy_vs_flat": 0.5,
            "fast_paced_vs_steady": 0.6,
            "results_vs_process": 0.6,
            "formal_vs_casual": 0.5,
            "risk_taking_vs_cautious": 0.0,
            "work_life_balance_vs_intensity": -0.5
        },
        "example_companies": ["McKinsey", "Deloitte", "Accenture", "Law firms"],
        "strengths": ["Variety", "Learning", "Network", "Prestige"],
        "challenges": ["Travel", "Long hours", "Client pressure", "Up-or-out culture"]
    },
    "remote_distributed": {
        "name": "Remote/Distributed",
        "description": "Fully remote or distributed teams. Values autonomy, async communication, and work-life balance.",
        "vector": {
            "innovation_vs_stability": 0.3,
            "collaboration_vs_independence": -0.4,
            "hierarchy_vs_flat": -0.5,
            "fast_paced_vs_steady": 0.1,
            "results_vs_process": 0.5,
            "formal_vs_casual": -0.5,
            "risk_taking_vs_cautious": 0.2,
            "work_life_balance_vs_intensity": 0.8
        },
        "example_companies": ["GitLab", "Automattic", "Zapier", "Buffer"],
        "strengths": ["Flexibility", "Autonomy", "Global talent", "Work-life balance"],
        "challenges": ["Isolation", "Communication", "Time zones", "Self-discipline needed"]
    },
    "family_business": {
        "name": "Family/Traditional Business",
        "description": "Close-knit, relationship-focused environments. Values loyalty, trust, and long-term thinking.",
        "vector": {
            "innovation_vs_stability": -0.5,
            "collaboration_vs_independence": 0.6,
            "hierarchy_vs_flat": 0.4,
            "fast_paced_vs_steady": -0.6,
            "results_vs_process": -0.2,
            "formal_vs_casual": 0.1,
            "risk_taking_vs_cautious": -0.5,
            "work_life_balance_vs_intensity": 0.5
        },
        "example_companies": ["Family-owned businesses", "Local enterprises", "Traditional industries"],
        "strengths": ["Relationships", "Stability", "Trust", "Personal touch"],
        "challenges": ["Limited growth", "Change resistance", "Informal practices", "Family dynamics"]
    },
    "research_academic": {
        "name": "Research/Academic",
        "description": "Knowledge-focused environments valuing deep expertise, intellectual rigor, and discovery.",
        "vector": {
            "innovation_vs_stability": 0.4,
            "collaboration_vs_independence": -0.2,
            "hierarchy_vs_flat": 0.2,
            "fast_paced_vs_steady": -0.7,
            "results_vs_process": -0.3,
            "formal_vs_casual": 0.3,
            "risk_taking_vs_cautious": 0.2,
            "work_life_balance_vs_intensity": 0.4
        },
        "example_companies": ["Universities", "Research labs", "Think tanks", "R&D centers"],
        "strengths": ["Deep expertise", "Intellectual freedom", "Contribution to knowledge", "Flexibility"],
        "challenges": ["Funding pressure", "Slow pace", "Limited practical application", "Academic politics"]
    },
    "healthcare_service": {
        "name": "Healthcare/Service",
        "description": "People-centered environments focused on care, service, and helping others. High empathy required.",
        "vector": {
            "innovation_vs_stability": -0.2,
            "collaboration_vs_independence": 0.7,
            "hierarchy_vs_flat": 0.3,
            "fast_paced_vs_steady": 0.3,
            "results_vs_process": 0.0,
            "formal_vs_casual": 0.4,
            "risk_taking_vs_cautious": -0.4,
            "work_life_balance_vs_intensity": -0.2
        },
        "example_companies": ["Hospitals", "Clinics", "Care facilities", "Health organizations"],
        "strengths": ["Meaningful impact", "Team environment", "Job security", "Helping others"],
        "challenges": ["Emotional demands", "Burnout", "Regulations", "Shift work"]
    },
    "government_public": {
        "name": "Government/Public Sector",
        "description": "Public service environments with structured processes. Values stability, service, and fairness.",
        "vector": {
            "innovation_vs_stability": -0.7,
            "collaboration_vs_independence": 0.3,
            "hierarchy_vs_flat": 0.8,
            "fast_paced_vs_steady": -0.7,
            "results_vs_process": -0.6,
            "formal_vs_casual": 0.8,
            "risk_taking_vs_cautious": -0.8,
            "work_life_balance_vs_intensity": 0.6
        },
        "example_companies": ["Government agencies", "Public institutions", "Municipal services"],
        "strengths": ["Job security", "Benefits", "Work-life balance", "Public service"],
        "challenges": ["Bureaucracy", "Slow processes", "Limited flexibility", "Political changes"]
    },
    "entrepreneurial_small": {
        "name": "Entrepreneurial/Small Business",
        "description": "Small teams with entrepreneurial spirit. Wears many hats, high ownership, direct impact on business.",
        "vector": {
            "innovation_vs_stability": 0.5,
            "collaboration_vs_independence": 0.2,
            "hierarchy_vs_flat": -0.7,
            "fast_paced_vs_steady": 0.5,
            "results_vs_process": 0.6,
            "formal_vs_casual": -0.6,
            "risk_taking_vs_cautious": 0.5,
            "work_life_balance_vs_intensity": -0.3
        },
        "example_companies": ["SMBs", "Local startups", "Boutique firms", "Owner-operated businesses"],
        "strengths": ["Autonomy", "Variety", "Direct impact", "Learning opportunities"],
        "challenges": ["Resource limits", "Wear many hats", "Less stability", "Limited benefits"]
    }
}


# =============================================================================
# Cultural Fit Service
# =============================================================================

class CulturalFitService:
    """
    Service for matching personality profiles to workplace culture types.

    Computes 8 culture dimensions from OCEAN scores and derived metrics,
    then uses cosine similarity to find best-matching culture types.
    """

    def __init__(self):
        self.derived_service = get_derived_metrics_service()
        logger.info("CulturalFitService initialized")

    def compute_culture_dimensions(
        self,
        ocean_scores: Dict[str, float],
        derived_metrics: Dict[str, Any] = None
    ) -> CultureDimensions:
        """
        Compute the 8 culture dimensions from personality scores.

        Args:
            ocean_scores: Dict with O, C, E, A, N scores (0-1 scale)
            derived_metrics: Optional pre-computed derived metrics

        Returns:
            CultureDimensions with all 8 dimension scores (-1 to +1)
        """
        # Normalize OCEAN scores
        O = self._normalize(ocean_scores.get('openness', 0.5))
        C = self._normalize(ocean_scores.get('conscientiousness', 0.5))
        E = self._normalize(ocean_scores.get('extraversion', 0.5))
        A = self._normalize(ocean_scores.get('agreeableness', 0.5))
        N = self._normalize(ocean_scores.get('neuroticism', 0.5))

        # Compute derived metrics if not provided
        if not derived_metrics:
            derived_metrics = {}

        # Helper to get metric score (normalized to 0-1)
        def get_metric(name: str, default: float = 0.5) -> float:
            if name in derived_metrics:
                m = derived_metrics[name]
                if isinstance(m, dict):
                    return m.get('score', default * 100) / 100
                return m / 100 if m > 1 else m
            return default

        # 1. Innovation vs Stability
        # High O + Low C + Risk tolerance → Innovation
        # Low O + High C + Low N → Stability
        innovation = (
            0.45 * (O - 0.5) * 2 +           # Openness drives innovation
            0.25 * (1 - C - 0.5) * 2 +        # Low C = less rigid
            0.20 * (E - 0.5) * 2 +            # E adds boldness
            0.10 * (1 - N - 0.5) * 2          # Stability for execution
        )
        innovation = max(-1, min(1, innovation))

        # 2. Collaboration vs Independence
        # High A + High E → Collaboration
        # Low A + Low E + High O → Independence
        collaboration = (
            0.40 * (A - 0.5) * 2 +            # Agreeableness = team player
            0.35 * (E - 0.5) * 2 +            # Extraversion = social energy
            0.25 * (1 - O - 0.5) * 2 * -1     # High O can mean independence
        )
        collaboration = max(-1, min(1, collaboration))

        # 3. Hierarchy vs Flat
        # High C + Low O → Hierarchy preference
        # High O + Low C + High E → Flat preference
        hierarchy = (
            0.40 * (C - 0.5) * 2 +            # C = follows structure
            0.30 * (1 - O - 0.5) * 2 +        # Low O = conventional
            0.20 * (1 - E - 0.5) * 2 +        # Low E = doesn't seek spotlight
            0.10 * (A - 0.5) * 2              # A = respects authority
        )
        hierarchy = max(-1, min(1, hierarchy))

        # 4. Fast-paced vs Steady
        # High E + Low N + High O → Fast-paced
        # Low E + High N + High C → Steady
        fast_paced = (
            0.35 * (E - 0.5) * 2 +            # E = energy
            0.30 * (1 - N - 0.5) * 2 +        # Low N = handles pressure
            0.20 * (O - 0.5) * 2 +            # O = adaptable
            0.15 * (1 - C - 0.5) * 2          # Low C = less methodical
        )
        fast_paced = max(-1, min(1, fast_paced))

        # 5. Results vs Process
        # High C + Low A (competitive) + High E → Results
        # High C + High A + Low E → Process
        results = (
            0.35 * (E - 0.5) * 2 +            # E = assertive
            0.30 * (1 - A - 0.5) * 2 +        # Low A = competitive
            0.20 * (C - 0.5) * 2 +            # C for execution
            0.15 * (1 - N - 0.5) * 2          # Low N = handles pressure
        )
        results = max(-1, min(1, results))

        # 6. Formal vs Casual
        # High C + Low O → Formal
        # Low C + High O + High E → Casual
        formal = (
            0.40 * (C - 0.5) * 2 +            # C = follows norms
            0.30 * (1 - O - 0.5) * 2 +        # Low O = conventional
            0.20 * (1 - E - 0.5) * 2 +        # Low E = reserved
            0.10 * (A - 0.5) * 2              # A = polite
        )
        formal = max(-1, min(1, formal))

        # 7. Risk-taking vs Cautious
        # High O + High E + Low N → Risk-taking
        # Low O + High N + High C → Cautious
        risk_taking = (
            0.35 * (O - 0.5) * 2 +            # O = seeks novelty
            0.30 * (E - 0.5) * 2 +            # E = bold
            0.25 * (1 - N - 0.5) * 2 +        # Low N = not anxious
            0.10 * (1 - C - 0.5) * 2          # Low C = less rule-bound
        )
        risk_taking = max(-1, min(1, risk_taking))

        # 8. Work-life Balance vs Intensity
        # High A + Low E + Moderate N → Balance
        # Low A + High E + Low N + High C → Intensity
        balance = (
            0.30 * (A - 0.5) * 2 +            # A = values relationships
            0.25 * (1 - E - 0.5) * 2 +        # Low E = less driven to achieve
            0.25 * (N - 0.5) * 2 +            # N = values self-care
            0.20 * (1 - C - 0.5) * 2          # Low C = less workaholic
        )
        balance = max(-1, min(1, balance))

        return CultureDimensions(
            innovation_vs_stability=innovation,
            collaboration_vs_independence=collaboration,
            hierarchy_vs_flat=hierarchy,
            fast_paced_vs_steady=fast_paced,
            results_vs_process=results,
            formal_vs_casual=formal,
            risk_taking_vs_cautious=risk_taking,
            work_life_balance_vs_intensity=balance
        )

    def _normalize(self, score: float) -> float:
        """Normalize score to 0-1 range"""
        if score < 0:
            return (score + 1) / 2
        return max(0, min(1, score))

    def _dimensions_to_vector(self, dims: CultureDimensions) -> List[float]:
        """Convert CultureDimensions to a list for similarity computation"""
        return [
            dims.innovation_vs_stability,
            dims.collaboration_vs_independence,
            dims.hierarchy_vs_flat,
            dims.fast_paced_vs_steady,
            dims.results_vs_process,
            dims.formal_vs_casual,
            dims.risk_taking_vs_cautious,
            dims.work_life_balance_vs_intensity
        ]

    def _culture_vector(self, culture_type: str) -> List[float]:
        """Get the vector for a culture type"""
        culture = CULTURE_TYPES.get(culture_type, {})
        vector = culture.get("vector", {})
        return [
            vector.get("innovation_vs_stability", 0),
            vector.get("collaboration_vs_independence", 0),
            vector.get("hierarchy_vs_flat", 0),
            vector.get("fast_paced_vs_steady", 0),
            vector.get("results_vs_process", 0),
            vector.get("formal_vs_casual", 0),
            vector.get("risk_taking_vs_cautious", 0),
            vector.get("work_life_balance_vs_intensity", 0)
        ]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def _get_profile_differentiation(self, person_vector: List[float]) -> float:
        """
        Measure how differentiated a profile is (magnitude of the vector).
        A balanced/moderate profile has low differentiation.
        Returns value between 0 (no differentiation) and 1 (max differentiation).
        """
        magnitude = math.sqrt(sum(v * v for v in person_vector))
        # Max possible magnitude is sqrt(8) ≈ 2.83 (if all dims are ±1)
        max_magnitude = math.sqrt(8)
        return min(magnitude / max_magnitude, 1.0)

    def _similarity_to_fit_score(
        self,
        similarity: float,
        differentiation: float = 1.0
    ) -> float:
        """
        Convert cosine similarity (-1 to 1) to fit score (0-100).
        Accounts for profile differentiation - moderate profiles get moderate scores.

        Args:
            similarity: Cosine similarity (-1 to 1)
            differentiation: How differentiated the person's profile is (0 to 1)
        """
        # Base score from similarity: map -1 to 0, 0 to 50, 1 to 100
        base_score = (similarity + 1) * 50

        # For low-differentiation profiles, compress scores toward 50%
        # A fully moderate profile (diff=0) should get ~50% for all cultures
        # A highly differentiated profile (diff=1) gets full range
        if differentiation < 0.3:
            # Very moderate profile: compress heavily toward 50
            compression = 0.3 + (differentiation / 0.3) * 0.4  # 0.3 to 0.7
        elif differentiation < 0.6:
            # Somewhat moderate: compress moderately
            compression = 0.7 + ((differentiation - 0.3) / 0.3) * 0.2  # 0.7 to 0.9
        else:
            # Differentiated profile: minimal compression
            compression = 0.9 + ((differentiation - 0.6) / 0.4) * 0.1  # 0.9 to 1.0

        # Apply compression: pull score toward 50
        compressed_score = 50 + (base_score - 50) * compression

        return round(compressed_score, 1)

    def get_culture_matches(
        self,
        ocean_scores: Dict[str, float],
        derived_metrics: Dict[str, Any] = None,
        top_n: int = 5
    ) -> List[CultureMatch]:
        """
        Get top culture type matches for a personality profile.

        Args:
            ocean_scores: Dict with O, C, E, A, N scores
            derived_metrics: Optional pre-computed derived metrics
            top_n: Number of top matches to return

        Returns:
            List of CultureMatch objects sorted by fit score
        """
        # Compute person's culture dimensions
        person_dims = self.compute_culture_dimensions(ocean_scores, derived_metrics)
        person_vector = self._dimensions_to_vector(person_dims)

        # Calculate profile differentiation (how distinct vs moderate the profile is)
        differentiation = self._get_profile_differentiation(person_vector)

        # Calculate similarity with each culture type
        matches = []
        for culture_key, culture_info in CULTURE_TYPES.items():
            culture_vector = self._culture_vector(culture_key)
            similarity = self._cosine_similarity(person_vector, culture_vector)
            fit_score = self._similarity_to_fit_score(similarity, differentiation)

            # Identify specific strengths for this person in this culture
            strengths = self._identify_strengths(person_dims, culture_key)
            challenges = self._identify_challenges(person_dims, culture_key)

            matches.append(CultureMatch(
                culture_type=culture_info["name"],
                fit_score=fit_score,
                description=culture_info["description"],
                strengths=strengths[:3],
                potential_challenges=challenges[:3],
                example_companies=culture_info["example_companies"]
            ))

        # Sort by fit score descending
        matches.sort(key=lambda m: m.fit_score, reverse=True)

        return matches[:top_n]

    def _identify_strengths(self, person_dims: CultureDimensions, culture_key: str) -> List[str]:
        """Identify specific strengths for this person in this culture"""
        culture = CULTURE_TYPES[culture_key]
        culture_vector = culture["vector"]
        strengths = []

        dimension_labels = {
            "innovation_vs_stability": ("innovation", "stability"),
            "collaboration_vs_independence": ("collaboration", "independence"),
            "hierarchy_vs_flat": ("structured environments", "flat organizations"),
            "fast_paced_vs_steady": ("fast-paced work", "steady environments"),
            "results_vs_process": ("results-driven work", "process-oriented work"),
            "formal_vs_casual": ("formal settings", "casual environments"),
            "risk_taking_vs_cautious": ("risk-taking", "careful planning"),
            "work_life_balance_vs_intensity": ("work-life balance", "intense focus")
        }

        person_vector = {
            "innovation_vs_stability": person_dims.innovation_vs_stability,
            "collaboration_vs_independence": person_dims.collaboration_vs_independence,
            "hierarchy_vs_flat": person_dims.hierarchy_vs_flat,
            "fast_paced_vs_steady": person_dims.fast_paced_vs_steady,
            "results_vs_process": person_dims.results_vs_process,
            "formal_vs_casual": person_dims.formal_vs_casual,
            "risk_taking_vs_cautious": person_dims.risk_taking_vs_cautious,
            "work_life_balance_vs_intensity": person_dims.work_life_balance_vs_intensity
        }

        for dim, (high_label, low_label) in dimension_labels.items():
            person_val = person_vector[dim]
            culture_val = culture_vector.get(dim, 0)

            # Check if person and culture align on this dimension
            if person_val > 0.3 and culture_val > 0.3:
                strengths.append(f"Your natural fit with {high_label}")
            elif person_val < -0.3 and culture_val < -0.3:
                strengths.append(f"Your preference for {low_label}")

        # Add general strengths from culture
        if not strengths:
            strengths = culture.get("strengths", [])[:2]

        return strengths

    def _identify_challenges(self, person_dims: CultureDimensions, culture_key: str) -> List[str]:
        """Identify potential challenges for this person in this culture"""
        culture = CULTURE_TYPES[culture_key]
        culture_vector = culture["vector"]
        challenges = []

        dimension_warnings = {
            "innovation_vs_stability": {
                "mismatch_high": "May feel constrained by the stability-focused environment",
                "mismatch_low": "May find the rapid change challenging"
            },
            "collaboration_vs_independence": {
                "mismatch_high": "May feel isolated in the independent work style",
                "mismatch_low": "May feel overwhelmed by constant collaboration"
            },
            "hierarchy_vs_flat": {
                "mismatch_high": "May struggle with the lack of clear structure",
                "mismatch_low": "May find the hierarchy restrictive"
            },
            "fast_paced_vs_steady": {
                "mismatch_high": "May feel the pace is too slow",
                "mismatch_low": "May feel overwhelmed by the fast pace"
            },
            "work_life_balance_vs_intensity": {
                "mismatch_high": "May struggle with the intense work demands",
                "mismatch_low": "May feel under-challenged by the relaxed pace"
            }
        }

        person_vector = {
            "innovation_vs_stability": person_dims.innovation_vs_stability,
            "collaboration_vs_independence": person_dims.collaboration_vs_independence,
            "hierarchy_vs_flat": person_dims.hierarchy_vs_flat,
            "fast_paced_vs_steady": person_dims.fast_paced_vs_steady,
            "work_life_balance_vs_intensity": person_dims.work_life_balance_vs_intensity
        }

        for dim, warnings in dimension_warnings.items():
            person_val = person_vector.get(dim, 0)
            culture_val = culture_vector.get(dim, 0)

            # Check for significant mismatch (person and culture on opposite ends)
            diff = person_val - culture_val
            if diff > 0.6:
                challenges.append(warnings["mismatch_high"])
            elif diff < -0.6:
                challenges.append(warnings["mismatch_low"])

        # Add general challenges from culture if no specific ones found
        if not challenges:
            challenges = culture.get("challenges", [])[:2]

        return challenges

    def get_cultural_fit_summary(
        self,
        ocean_scores: Dict[str, float],
        derived_metrics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get a comprehensive cultural fit analysis.

        Args:
            ocean_scores: Dict with O, C, E, A, N scores
            derived_metrics: Optional pre-computed derived metrics

        Returns:
            Dict with dimensions, top matches, and recommendations
        """
        # Compute dimensions
        dims = self.compute_culture_dimensions(ocean_scores, derived_metrics)

        # Get top matches
        matches = self.get_culture_matches(ocean_scores, derived_metrics, top_n=5)

        # Format dimensions for output
        dimension_labels = {
            "innovation_vs_stability": "Innovation ↔ Stability",
            "collaboration_vs_independence": "Collaboration ↔ Independence",
            "hierarchy_vs_flat": "Hierarchical ↔ Flat",
            "fast_paced_vs_steady": "Fast-paced ↔ Steady",
            "results_vs_process": "Results ↔ Process",
            "formal_vs_casual": "Formal ↔ Casual",
            "risk_taking_vs_cautious": "Risk-taking ↔ Cautious",
            "work_life_balance_vs_intensity": "Work-life Balance ↔ Intensity"
        }

        formatted_dims = {}
        for attr, label in dimension_labels.items():
            value = getattr(dims, attr)
            formatted_dims[label] = {
                "value": round(value, 2),
                "tendency": self._describe_tendency(attr, value)
            }

        # Format matches
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                "culture_type": match.culture_type,
                "fit_score": match.fit_score,
                "description": match.description,
                "strengths": match.strengths,
                "potential_challenges": match.potential_challenges,
                "example_companies": match.example_companies
            })

        # Calculate profile differentiation
        person_vector = self._dimensions_to_vector(dims)
        differentiation = self._get_profile_differentiation(person_vector)

        # Determine profile type
        if differentiation < 0.25:
            profile_type = "Very Balanced/Adaptable"
            profile_note = "Your scores are moderate across all traits, giving you flexibility in many environments."
        elif differentiation < 0.4:
            profile_type = "Moderately Balanced"
            profile_note = "You have some distinct preferences but remain adaptable."
        else:
            profile_type = "Differentiated"
            profile_note = "You have clear preferences that point toward specific culture types."

        return {
            "dimensions": formatted_dims,
            "top_matches": formatted_matches,
            "best_fit": formatted_matches[0] if formatted_matches else None,
            "profile_type": profile_type,
            "profile_note": profile_note,
            "differentiation": round(differentiation * 100, 1),
            "recommendation": self._generate_recommendation(matches, dims)
        }

    def _describe_tendency(self, dimension: str, value: float) -> str:
        """Describe the tendency based on dimension value"""
        tendencies = {
            "innovation_vs_stability": ("Stability-oriented", "Balanced", "Innovation-seeking"),
            "collaboration_vs_independence": ("Independence-focused", "Flexible", "Collaboration-oriented"),
            "hierarchy_vs_flat": ("Prefers flat structures", "Adaptable", "Comfortable with hierarchy"),
            "fast_paced_vs_steady": ("Prefers steady pace", "Adaptable", "Thrives in fast-paced"),
            "results_vs_process": ("Process-focused", "Balanced", "Results-driven"),
            "formal_vs_casual": ("Prefers casual", "Adaptable", "Comfortable with formal"),
            "risk_taking_vs_cautious": ("Cautious approach", "Balanced", "Comfortable with risk"),
            "work_life_balance_vs_intensity": ("Intensity-focused", "Flexible", "Values balance")
        }

        labels = tendencies.get(dimension, ("Low", "Moderate", "High"))

        if value < -0.3:
            return labels[0]
        elif value > 0.3:
            return labels[2]
        else:
            return labels[1]

    def _generate_recommendation(self, matches: List[CultureMatch], dims: CultureDimensions) -> str:
        """Generate a personalized recommendation based on matches and dimensions"""
        if not matches:
            return "Complete personality assessment to get cultural fit recommendations."

        top_match = matches[0]
        person_vector = self._dimensions_to_vector(dims)
        differentiation = self._get_profile_differentiation(person_vector)

        # Build recommendation
        parts = []

        # Check if profile is moderate/balanced
        if differentiation < 0.25:
            parts.append("Your personality profile is **well-balanced and adaptable**, without strong leanings in any particular direction.")
            parts.append("This means you can likely fit reasonably well in many different workplace cultures.")
            parts.append(f"Your slight tendencies align best with **{top_match.culture_type}** environments ({top_match.fit_score}% fit).")
            parts.append("Consider what aspects of work culture matter most to YOU personally, since your natural adaptability gives you flexibility.")
        elif differentiation < 0.4:
            parts.append(f"Your personality shows some preferences that align with **{top_match.culture_type}** environments ({top_match.fit_score}% fit).")
            if len(matches) > 1 and matches[1].fit_score > 55:
                parts.append(f"You'd also be comfortable in **{matches[1].culture_type}** settings.")
            parts.append("Your relatively balanced profile gives you adaptability across different environments.")
        else:
            parts.append(f"Your personality profile aligns strongly with **{top_match.culture_type}** environments ({top_match.fit_score}% fit).")
            if len(matches) > 1 and matches[1].fit_score > 65:
                parts.append(f"You'd also thrive in **{matches[1].culture_type}** settings.")

        # Add dimension-specific insights only for differentiated profiles
        if differentiation >= 0.3:
            if dims.innovation_vs_stability > 0.5:
                parts.append("Your innovative nature will be valued in dynamic, change-oriented organizations.")
            elif dims.innovation_vs_stability < -0.5:
                parts.append("Your preference for stability makes you well-suited for established organizations with clear processes.")

            if dims.collaboration_vs_independence < -0.3:
                parts.append("Look for roles with autonomy and independent project ownership.")
            elif dims.collaboration_vs_independence > 0.3:
                parts.append("Seek team-oriented roles where collaboration is central.")

        return " ".join(parts)


# =============================================================================
# Tool Function for AI Agent
# =============================================================================

def get_cultural_fit_analysis(
    ocean_scores: Dict[str, float],
    derived_metrics: Dict[str, Any] = None
) -> str:
    """
    Get cultural fit analysis for the AI agent tool.

    Args:
        ocean_scores: Dict with O, C, E, A, N scores
        derived_metrics: Optional derived metrics

    Returns:
        Formatted string with cultural fit analysis
    """
    service = CulturalFitService()
    result = service.get_cultural_fit_summary(ocean_scores, derived_metrics)

    # Format for agent response
    lines = ["**Cultural Fit Analysis**\n"]

    # Profile type indicator
    lines.append(f"**Profile Type:** {result['profile_type']} (differentiation: {result['differentiation']}%)")
    lines.append(f"_{result['profile_note']}_\n")

    # Top matches
    lines.append("**Best Culture Matches:**")
    for i, match in enumerate(result["top_matches"][:3], 1):
        lines.append(f"\n{i}. **{match['culture_type']}** ({match['fit_score']}% fit)")
        lines.append(f"   {match['description']}")
        if match['strengths']:
            lines.append(f"   ✓ Strengths: {', '.join(match['strengths'][:2])}")
        if match['potential_challenges']:
            lines.append(f"   ⚠ Watch for: {', '.join(match['potential_challenges'][:2])}")
        if match['example_companies']:
            lines.append(f"   📍 Examples: {', '.join(match['example_companies'][:3])}")

    # Recommendation
    lines.append(f"\n**Recommendation:**\n{result['recommendation']}")

    return "\n".join(lines)


# Singleton instance
_cultural_fit_service: CulturalFitService = None


def get_cultural_fit_service() -> CulturalFitService:
    """Get or create singleton cultural fit service"""
    global _cultural_fit_service
    if _cultural_fit_service is None:
        _cultural_fit_service = CulturalFitService()
    return _cultural_fit_service
