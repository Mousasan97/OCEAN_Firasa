"""
Reference norms for OCEAN personality traits
Computed from train_single_image_annotations_CLEANED.csv
"""

# Reference norms (mean and std) from training data
REFERENCE_NORMS = {
    "extraversion": {
        "mean": 0.476364,
        "std": 0.152412
    },
    "neuroticism": {
        "mean": 0.520432,
        "std": 0.153773
    },
    "agreeableness": {
        "mean": 0.548365,
        "std": 0.136434
    },
    "conscientiousness": {
        "mean": 0.523032,
        "std": 0.155261
    },
    "openness": {
        "mean": 0.566469,
        "std": 0.147068
    }
}

# T-score category thresholds (NEO/BFI-aligned)
T_SCORE_THRESHOLDS = {
    "very_low": 30,      # T ≤ 30
    "low": 44,           # 31 ≤ T ≤ 44
    "average": 55,       # 45 ≤ T ≤ 55
    "high": 64,          # 56 ≤ T ≤ 64
    "very_high": 65      # T ≥ 65
}

# Validated narrative interpretations (NEO-PI-R / BFI-2 / IPIP-NEO aligned)
TRAIT_INTERPRETATIONS = {
    "extraversion": {
        "Very Low": "Reserved, quiet, prefers solitude; gains energy from reflection.",
        "Low": "Moderately quiet; engages socially when required but enjoys independence.",
        "Average": "Balanced between sociability and introspection; comfortable alone or with others.",
        "High": "Outgoing, energetic, expressive; enjoys group work and discussion.",
        "Very High": "Highly sociable, talkative, enthusiastic; thrives on interaction and excitement."
    },
    "neuroticism": {
        "Very Low": "Calm and emotionally stable; rarely upset even under pressure.",
        "Low": "Generally composed; occasional tension but recovers quickly.",
        "Average": "Typical emotional balance; sometimes stressed, usually resilient.",
        "High": "Emotionally sensitive and self-aware; may worry under uncertainty.",
        "Very High": "Strong emotional reactivity; benefits from structure and coping routines."
    },
    "agreeableness": {
        "Very Low": "Tough-minded, frank, skeptical; prioritizes logic over harmony.",
        "Low": "Direct and sometimes critical; cooperative when goals align.",
        "Average": "Kind and fair; balances assertiveness with empathy.",
        "High": "Warm, considerate, empathetic; seeks harmony in relationships.",
        "Very High": "Exceptionally trusting and altruistic; may overextend to help others."
    },
    "conscientiousness": {
        "Very Low": "Spontaneous and flexible; dislikes strict routines or detailed planning.",
        "Low": "Casual and adaptable; may overlook details or deadlines at times.",
        "Average": "Reasonably organized and dependable; keeps most commitments.",
        "High": "Reliable, disciplined, responsible; sets and pursues clear goals.",
        "Very High": "Exceptionally orderly and perfectionistic; highly self-demanding and driven."
    },
    "openness": {
        "Very Low": "Conventional, prefers familiar methods and traditions.",
        "Low": "Practical and realistic; curious when necessary but values routine.",
        "Average": "Balances imagination with practicality; open when relevant.",
        "High": "Curious, creative, appreciates ideas and art; enjoys novelty.",
        "Very High": "Highly imaginative and visionary; embraces complexity and change."
    }
}

# Category-consistent short labels (don't contradict the category)
TRAIT_LABELS = {
    "extraversion": {
        "Very Low": "Quiet, prefers solitude",
        "Low": "Reserved, selectively social",
        "Average": "Balanced sociability",
        "High": "Outgoing, energetic",
        "Very High": "Highly sociable and expressive"
    },
    "neuroticism": {
        "Very Low": "Very calm, unflappable",
        "Low": "Composed, steady",
        "Average": "Emotionally balanced",
        "High": "Sensitive to stress",
        "Very High": "Highly reactive to stress"
    },
    "agreeableness": {
        "Very Low": "Tough-minded, frank",
        "Low": "Direct, occasional friction",
        "Average": "Cooperative, fair",
        "High": "Warm, empathetic",
        "Very High": "Highly altruistic"
    },
    "conscientiousness": {
        "Very Low": "Spontaneous, unstructured",
        "Low": "Casual, flexible",
        "Average": "Dependable, organized",
        "High": "Disciplined, goal-driven",
        "Very High": "Meticulous, perfection-oriented"
    },
    "openness": {
        "Very Low": "Conventional, routine-oriented",
        "Low": "Practical, familiar solutions",
        "Average": "Balanced curiosity",
        "High": "Curious, imaginative",
        "Very High": "Visionary, experimental"
    }
}

# Metadata for reproducibility
NORMS_METADATA = {
    "source": "train_single_image_annotations_CLEANED.csv",
    "computed_at": "2025-01-15",
    "dataset_size": None,  # To be filled if known
    "version": "1.0"
}
