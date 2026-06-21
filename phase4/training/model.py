"""
ShortagePredictor — PyTorch model for inventory shortage probability.

Architecture:
  warehouse_id → Embedding(n_wh, wh_dim)  ┐
  sku_id       → Embedding(n_sku, sku_dim) ┤ concatenated
  numeric      → (pre-scaled externally)   ┘
                         ↓
          MLP trunk: [Linear → BatchNorm1d → ReLU → Dropout] × N layers
                         ↓
          prob_head  → logit  (BCEWithLogitsLoss during training)
          score_head → sigmoid (MSELoss against continuous_risk_score)

forward() returns (logit_prob, score):
  logit_prob — raw logit, feed directly to BCEWithLogitsLoss
  score      — sigmoid output ∈ [0, 1], feed to MSELoss

predict() returns (prob, score) — both ∈ [0, 1], for inference use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from phase4.training.config import TrainingConfig


class ShortagePredictor(nn.Module):

    def __init__(
        self,
        n_numeric:    int,
        n_warehouses: int,
        n_skus:       int,
        config:       TrainingConfig,
    ) -> None:
        super().__init__()

        self.wh_embedding  = nn.Embedding(n_warehouses, config.wh_embedding_dim)
        self.sku_embedding = nn.Embedding(n_skus,       config.sku_embedding_dim)

        in_dim = n_numeric + config.wh_embedding_dim + config.sku_embedding_dim
        layers: list[nn.Module] = []
        for out_dim, drop in zip(config.hidden_dims, config.dropout_rates):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if drop > 0.0:
                layers.append(nn.Dropout(p=drop))
            in_dim = out_dim

        self.trunk      = nn.Sequential(*layers)
        self.prob_head  = nn.Linear(in_dim, 1)
        self.score_head = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(
        self,
        x_numeric: torch.Tensor,
        wh_ids:    torch.Tensor,
        sku_ids:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_numeric : (batch, n_numeric)  — pre-scaled float features
        wh_ids    : (batch,)             — warehouse index ∈ [0, n_warehouses)
        sku_ids   : (batch,)             — sku index ∈ [0, n_skus)

        Returns
        -------
        logit_prob : (batch, 1) — raw logit for BCEWithLogitsLoss
        score      : (batch, 1) — sigmoid risk score for MSELoss
        """
        x = torch.cat(
            [x_numeric, self.wh_embedding(wh_ids), self.sku_embedding(sku_ids)],
            dim=1,
        )
        h = self.trunk(x)
        return self.prob_head(h), torch.sigmoid(self.score_head(h))

    @torch.no_grad()
    def predict(
        self,
        x_numeric: torch.Tensor,
        wh_ids:    torch.Tensor,
        sku_ids:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference-mode forward. Returns (prob, score) both ∈ [0, 1]."""
        self.eval()
        logit, score = self.forward(x_numeric, wh_ids, sku_ids)
        return torch.sigmoid(logit), score
