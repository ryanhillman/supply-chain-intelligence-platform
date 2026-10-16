"""
Loss functions for ShortagePredictor.

WeightedBCELoss — BCEWithLogitsLoss with class-imbalance pos_weight.
FocalLoss       — down-weights easy examples; useful when positive rate > 85%.
CombinedLoss    — binary_loss + risk_score_weight × MSELoss on continuous target.
build_loss()    — factory that reads TrainingConfig and returns CombinedLoss.
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


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al. 2017).
    Down-weights the loss contribution of easy well-classified examples
    so training focuses on hard boundary cases.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.where(targets == 1.0, torch.sigmoid(logits), 1.0 - torch.sigmoid(logits))
        return (((1.0 - pt) ** self.gamma) * bce).mean()


class CombinedLoss(nn.Module):
    """
    total = binary_loss(logit_prob, stockout_label_7d)
          + risk_score_weight × MSE(pred_score, continuous_risk_score)

    Returns (total, binary_component, score_component) so callers can log
    each term separately.
    """
    def __init__(self, binary_loss: nn.Module, risk_score_weight: float = 0.3) -> None:
        super().__init__()
        self.binary_loss       = binary_loss
        self.risk_score_weight = risk_score_weight

    def forward(
        self,
        logit_prob:   torch.Tensor,
        pred_score:   torch.Tensor,
        label_binary: torch.Tensor,
        label_risk:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_b = self.binary_loss(logit_prob, label_binary)
        loss_s = F.mse_loss(pred_score, label_risk)
        total  = loss_b + self.risk_score_weight * loss_s
        return total, loss_b, loss_s


def build_loss(config: TrainingConfig, pos_weight: float) -> CombinedLoss:
    """
    Construct the correct binary loss from config and wrap in CombinedLoss.

    pos_weight is computed by DatasetBuilder from the training split:
        pos_weight = n_negative / n_positive
    """
    if config.loss_type == "focal":
        binary: nn.Module = FocalLoss(gamma=config.focal_gamma, pos_weight=pos_weight)
    else:
        binary = WeightedBCELoss(pos_weight=pos_weight)
    return CombinedLoss(binary, risk_score_weight=config.risk_score_weight)
