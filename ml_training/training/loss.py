"""
Loss function for ShortagePredictor.

WeightedBCELoss — BCEWithLogitsLoss with an optional class-imbalance pos_weight
                  (pos_weight=1.0 is plain BCE, the default).
build_loss()    — factory that reads TrainingConfig.loss_type. The model has a
                  single logit output; stockout_risk_score is a closed-form
                  formula at scoring time, not a trained head, so there is no
                  auxiliary regression term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_training.training.config import TrainingConfig


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def build_loss(config: TrainingConfig, pos_weight: float) -> nn.Module:
    """
    Construct the binary loss selected by config.loss_type.

    pos_weight is computed by DatasetBuilder from the training split:
        pos_weight = n_negative / n_positive
    It is applied only for loss_type "weighted_bce".
    """
    if config.loss_type == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    return WeightedBCELoss(pos_weight=1.0)   # "bce" — unweighted; dataset pos_weight ignored
