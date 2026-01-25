"""
Personality interpretation service
Converts raw OCEAN scores to T-scores and validated narrative interpretations
"""
from typing import Dict, Any
from dataclasses import dataclass

from src.core.reference_norms import (
    REFERENCE_NORMS,
    T_SCORE_THRESHOLDS,
    TRAIT_INTERPRETATIONS,
    TRAIT_LABELS,
    NORMS_METADATA
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TraitInterpretation:
    """Individual trait interpretation"""
    trait: str
    raw_score: float
    t_score: float
    percentile: float
    category: str
    label: str
    interpretation: str


class InterpretationService:
    """
    Service for interpreting OCEAN personality scores
    Implements T-score transformation and validated narrative mapping
    """

    def __init__(self):
        """Initialize interpretation service"""
        self.norms = REFERENCE_NORMS
        self.thresholds = T_SCORE_THRESHOLDS
        self.narratives = TRAIT_INTERPRETATIONS
        self.labels = TRAIT_LABELS
        self.norms_metadata = NORMS_METADATA

        logger.info("InterpretationService initialized")

    @staticmethod
    def clip_score(score: float) -> float:
        """
        Clip score to valid range [0, 1]

        Args:
            score: Raw score

        Returns:
            Clipped score
        """
        return min(max(score, 0.0), 1.0)

    def compute_t_score(self, raw_score: float, trait: str) -> float:
        """
        Compute T-score from raw score using reference norms

        T-score formula: T = 50 + 10 * ((x - μ) / σ)

        Args:
            raw_score: Raw score in [0, 1]
            trait: Trait name

        Returns:
            T-score (typically in range ~20-80, mean=50, std=10)
        """
        if trait not in self.norms:
            logger.warning(f"Unknown trait: {trait}, using default norms")
            mean, std = 0.5, 0.15  # Fallback
        else:
            mean = self.norms[trait]["mean"]
            std = self.norms[trait]["std"]

        # Clip raw score to [0, 1]
        clipped_score = self.clip_score(raw_score)

        # Compute T-score
        t_score = 50 + 10 * ((clipped_score - mean) / std)

        return round(t_score, 2)

    @staticmethod
    def compute_percentile(t_score: float) -> float:
        """
        Compute percentile from T-score using standard normal distribution

        T-scores have mean=50, std=10
        percentile = Φ((T-50)/10) * 100

        Args:
            t_score: T-score value

        Returns:
            Percentile (0-100)
        """
        from scipy.stats import norm
        z_score = (t_score - 50) / 10
        percentile = norm.cdf(z_score) * 100
        return round(percentile, 1)

    def categorize_t_score(self, t_score: float) -> str:
        """
        Categorize T-score into 5 levels

        Categories (NEO/BFI-aligned):
        - Very Low: T ≤ 30
        - Low: 31 ≤ T ≤ 44
        - Average: 45 ≤ T ≤ 55
        - High: 56 ≤ T ≤ 64
        - Very High: T ≥ 65

        Args:
            t_score: T-score value

        Returns:
            Category string
        """
        if t_score <= self.thresholds["very_low"]:
            return "Very Low"
        elif t_score <= self.thresholds["low"]:
            return "Low"
        elif t_score <= self.thresholds["average"]:
            return "Average"
        elif t_score <= self.thresholds["high"]:
            return "High"
        else:
            return "Very High"

    def get_narrative(self, trait: str, category: str) -> str:
        """
        Get validated narrative interpretation for trait and category

        Args:
            trait: Trait name
            category: Category (Very Low, Low, Average, High, Very High)

        Returns:
            Narrative interpretation text
        """
        if trait not in self.narratives:
            logger.warning(f"Unknown trait for narrative: {trait}")
            return "No interpretation available."

        if category not in self.narratives[trait]:
            logger.warning(f"Unknown category for {trait}: {category}")
            return "No interpretation available."

        return self.narratives[trait][category]

    def get_label(self, trait: str, category: str) -> str:
        """
        Get category-consistent short label

        Args:
            trait: Trait name
            category: Category (Very Low, Low, Average, High, Very High)

        Returns:
            Short label text
        """
        if trait not in self.labels:
            logger.warning(f"Unknown trait for label: {trait}")
            return category

        if category not in self.labels[trait]:
            logger.warning(f"Unknown category for {trait}: {category}")
            return category

        return self.labels[trait][category]

    def interpret_trait(
        self,
        trait: str,
        raw_score: float
    ) -> TraitInterpretation:
        """
        Complete interpretation pipeline for a single trait

        Args:
            trait: Trait name
            raw_score: Raw score from model

        Returns:
            TraitInterpretation object
        """
        # Clip raw score for output
        clipped_score = self.clip_score(raw_score)

        # Compute T-score
        t_score = self.compute_t_score(clipped_score, trait)

        # Compute percentile
        percentile = self.compute_percentile(t_score)

        # Categorize
        category = self.categorize_t_score(t_score)

        # Get category-consistent label
        label = self.get_label(trait, category)

        # Get narrative interpretation
        interpretation = self.get_narrative(trait, category)

        return TraitInterpretation(
            trait=trait,
            raw_score=round(clipped_score, 4),  # Round to 4 decimals
            t_score=t_score,  # Already rounded to 2 decimals
            percentile=percentile,
            category=category,
            label=label,
            interpretation=interpretation
        )

    def interpret_all_traits(
        self,
        predictions: Dict[str, float]
    ) -> Dict[str, TraitInterpretation]:
        """
        Interpret all OCEAN traits

        Args:
            predictions: Dictionary of trait predictions
                         {trait: raw_score, ...}

        Returns:
            Dictionary of trait interpretations
            {trait: TraitInterpretation, ...}
        """
        interpretations = {}

        for trait, raw_score in predictions.items():
            interpretation = self.interpret_trait(trait, raw_score)
            interpretations[trait] = interpretation

        logger.info(f"Interpreted {len(interpretations)} traits")
        return interpretations

    def format_interpretation_dict(
        self,
        interpretations: Dict[str, TraitInterpretation]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Format interpretations as JSON-serializable dictionary

        Args:
            interpretations: Dictionary of TraitInterpretation objects

        Returns:
            Formatted dictionary
        """
        result = {}

        for trait, interp in interpretations.items():
            result[trait] = {
                "raw_score": interp.raw_score,
                "t_score": interp.t_score,
                "percentile": interp.percentile,
                "category": interp.category,
                "label": interp.label,
                "interpretation": interp.interpretation
            }

        return result

    def get_summary(
        self,
        interpretations: Dict[str, TraitInterpretation]
    ) -> Dict[str, Any]:
        """
        Get overall personality summary

        Args:
            interpretations: Dictionary of TraitInterpretation objects

        Returns:
            Summary dictionary
        """
        # Count categories
        category_counts = {
            "Very Low": 0,
            "Low": 0,
            "Average": 0,
            "High": 0,
            "Very High": 0
        }

        for interp in interpretations.values():
            category_counts[interp.category] += 1

        # Find dominant traits (T ≥ 60, or stricter: T ≥ 65)
        dominant_traits = [
            interp.trait
            for interp in interpretations.values()
            if interp.t_score >= 60  # Can use 65 for stricter threshold
        ]

        # Find subdued traits (T ≤ 40, or stricter: T ≤ 35)
        subdued_traits = [
            interp.trait
            for interp in interpretations.values()
            if interp.t_score <= 40  # Can use 35 for stricter threshold
        ]

        # Compute mean T-score
        mean_t_score = sum(
            interp.t_score for interp in interpretations.values()
        ) / len(interpretations) if interpretations else 50.0

        return {
            "category_distribution": category_counts,
            "dominant_traits": dominant_traits,
            "subdued_traits": subdued_traits,
            "mean_t_score": round(mean_t_score, 2),
            "total_traits": len(interpretations)
        }

    def get_norms_metadata(self) -> Dict[str, Any]:
        """
        Get norms metadata for reproducibility

        Returns:
            Metadata dictionary with norms information
        """
        return {
            "norms": {
                "source": self.norms_metadata["source"],
                "computed_at": self.norms_metadata["computed_at"],
                "version": self.norms_metadata["version"],
                "means": {trait: data["mean"] for trait, data in self.norms.items()},
                "stds": {trait: data["std"] for trait, data in self.norms.items()}
            },
            "cutoffs": {
                "very_low": self.thresholds["very_low"],
                "low_hi": self.thresholds["low"],
                "avg_hi": self.thresholds["average"],
                "high_hi": self.thresholds["high"],
                "very_high": self.thresholds["very_high"]
            }
        }


# Global instance
_interpretation_service = None


def get_interpretation_service() -> InterpretationService:
    """Get interpretation service instance (singleton)"""
    global _interpretation_service
    if _interpretation_service is None:
        _interpretation_service = InterpretationService()
    return _interpretation_service
