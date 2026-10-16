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
    # "weighted_bce" — BCEWithLogitsLoss(pos_weight=n_neg/n_pos)
    # "focal"        — FocalLoss with focal_gamma
    loss_type:         str   = "weighted_bce"
    focal_gamma:       float = 2.0
    risk_score_weight: float = 0.3   # coefficient on the MSE risk-score term

    # ── LR scheduler ─────────────────────────────────────────────────────────
    # "cosine"  — CosineAnnealingLR(T_max=T_max)
    # "plateau" — ReduceLROnPlateau monitoring val AUPRC
    scheduler:   str   = "cosine"
    T_max:       int   = 20
    lr_factor:   float = 0.5
    lr_patience: int   = 5

    # ── Paths ─────────────────────────────────────────────────────────────────
    dataset_dir: str = "ml_training/artifacts/dataset"
    output_dir:  str = "ml_training/artifacts"

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
