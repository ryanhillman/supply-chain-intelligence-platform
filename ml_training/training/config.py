"""
Training configuration — all hyperparameters and path constants.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # ── Model architecture ────────────────────────────────────────────────────
    hidden_dims:       tuple[int, ...]   = (64, 32, 16)
    dropout_rates:     tuple[float, ...] = (0.3, 0.2, 0.0)
    wh_embedding_dim:  int   = 3
    sku_embedding_dim: int   = 5

    # ── Optimization ──────────────────────────────────────────────────────────
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    batch_size:   int   = 64
    max_epochs:   int   = 100
    patience:     int   = 10     # early-stopping patience on val AUPRC

    # ── Loss ──────────────────────────────────────────────────────────────────
    # "bce"          — BCEWithLogitsLoss, unweighted (default)
    # "weighted_bce" — BCEWithLogitsLoss(pos_weight=n_neg/n_pos)
    # Unweighted is the default: on the seed-42 synthetic split, pos_weight
    # (3.59) moved test AUPRC 0.732 → 0.719 and Brier 0.076 → 0.117. The
    # weight inflates every predicted probability, which hurts calibration
    # without improving ranking.
    loss_type:         str   = "bce"

    # ── LR scheduler ─────────────────────────────────────────────────────────
    # "cosine"  — CosineAnnealingLR over T_max epochs; None means max_epochs,
    #             i.e. one anneal from lr down to 1e-6 across the whole run
    # "plateau" — ReduceLROnPlateau monitoring val AUPRC
    scheduler:   str        = "cosine"
    T_max:       int | None = None
    lr_factor:   float      = 0.5
    lr_patience: int        = 5

    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_dir: str = "ml_training/artifacts/dataset"
    output_dir:  str = "ml_training/artifacts"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
