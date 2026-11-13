"""Thresholds shared across model_serving: the variance-collapse threshold and
its check (the scorer, the drift monitor and the risk dashboard) and the two
dashboards' high-risk cutoff."""

# A std of shortage_prob_7d below this value indicates
# the model has lost discriminative power and retraining is required.
VARIANCE_COLLAPSE_THRESHOLD: float = 0.01

# The shortage_prob_7d cutoff both dashboard scripts use for "high risk".
HIGH_RISK_THRESHOLD: float = 0.50


def variance_collapsed(std: float) -> bool:
    """True when the std of shortage_prob_7d is below VARIANCE_COLLAPSE_THRESHOLD."""
    return std < VARIANCE_COLLAPSE_THRESHOLD
