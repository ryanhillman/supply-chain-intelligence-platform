"""The variance-collapse threshold and its check, shared by the scorer, the drift
monitor and the risk dashboard."""

# A std of shortage_prob_7d below this value indicates
# the model has lost discriminative power and retraining is required.
VARIANCE_COLLAPSE_THRESHOLD: float = 0.01


def variance_collapsed(std: float) -> bool:
    """True when the std of shortage_prob_7d is below VARIANCE_COLLAPSE_THRESHOLD."""
    return std < VARIANCE_COLLAPSE_THRESHOLD
