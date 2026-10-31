"""Shared constants for Phase 5 pipeline components."""

# A std of shortage_prob_7d below this value indicates
# the model has lost discriminative power and retraining is required.
VARIANCE_COLLAPSE_THRESHOLD: float = 0.01
