"""
ShortagePredictor — PyTorch model for inventory shortage probability.

Architecture:
  numeric      → Standardize((x − μ) / σ; μ, σ fit on the train split) ┐
  warehouse_id → Embedding(n_wh, wh_dim)                                 ┤ concatenated
  sku_id       → Embedding(n_sku, sku_dim)                               ┘
                         ↓
          MLP trunk: [Linear → BatchNorm1d → ReLU → Dropout] × N layers
                         ↓
          prob_head  → logit  (BCEWithLogitsLoss during training)
          score_head → sigmoid (MSELoss against continuous_risk_score)

The model owns standardization. μ and σ are registered buffers — saved in the
checkpoint and exported inside the ONNX graph — so every consumer (trainer,
evaluator, ONNX export, serving container, batch scorer) feeds raw feature
units. See ADR-005 Decision 3.

forward() returns (logit_prob, score):
  logit_prob — raw logit, feed directly to BCEWithLogitsLoss
  score      — sigmoid output ∈ [0, 1], feed to MSELoss

predict() returns (prob, score) — both ∈ [0, 1], for inference use.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ml_training.training.config import TrainingConfig


class Standardize(nn.Module):
    """
    Fixed (x − mean) / scale over the numeric feature columns.

    mean and scale are buffers, not parameters: DatasetBuilder fits them once
    on the training split, they travel with state_dict and the ONNX export,
    and the optimizer never sees them (AdamW's weight decay would otherwise
    drift them). scale must already carry sklearn's zero-variance guard
    (1.0 where the training std is 0), which DatasetBuilder provides by
    taking the values from a fitted StandardScaler.
    """

    def __init__(self, mean: Sequence[float], scale: Sequence[float]) -> None:
        super().__init__()
        mean_t  = torch.as_tensor(list(mean),  dtype=torch.float32)
        scale_t = torch.as_tensor(list(scale), dtype=torch.float32)
        if mean_t.ndim != 1 or mean_t.shape != scale_t.shape:
            raise ValueError(
                "mean and scale must be 1-D and the same length, got "
                f"{tuple(mean_t.shape)} and {tuple(scale_t.shape)}"
            )
        if bool(torch.any(scale_t <= 0)):
            raise ValueError(
                "scale must be strictly positive (sklearn maps zero variance to 1.0)"
            )
        self.register_buffer("mean",  mean_t)
        self.register_buffer("scale", scale_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.scale


class ShortagePredictor(nn.Module):

    def __init__(
        self,
        n_numeric:     int,
        n_warehouses:  int,
        n_skus:        int,
        config:        TrainingConfig,
        feature_mean:  Sequence[float],
        feature_scale: Sequence[float],
    ) -> None:
        super().__init__()

        if len(feature_mean) != n_numeric:
            raise ValueError(
                f"feature_mean has {len(feature_mean)} entries but n_numeric is {n_numeric}"
            )
        self.standardize = Standardize(feature_mean, feature_scale)

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

    @classmethod
    def from_metadata(cls, metadata: dict, config: TrainingConfig) -> "ShortagePredictor":
        """
        Build from feature_metadata.json (written by DatasetBuilder): feature
        count, vocab sizes, and the train-split standardization statistics.
        Callers that load a checkpoint afterwards get the buffers overwritten
        by the saved state_dict, which holds the same values.
        """
        return cls(
            n_numeric     = metadata["n_numeric_features"],
            n_warehouses  = metadata["n_warehouses"],
            n_skus        = metadata["n_skus"],
            config        = config,
            feature_mean  = metadata["feature_mean"],
            feature_scale = metadata["feature_scale"],
        )

    def _init_weights(self) -> None:
        # prob_head and score_head are final linear outputs with no
        # following ReLU (prob_head feeds BCEWithLogitsLoss on the raw
        # logit; score_head's sigmoid is applied in forward(), not ReLU) —
        # Kaiming's relu gain (sqrt(2)) doesn't apply to them, so they're
        # initialized separately with the linear gain (1.0).
        output_heads = {self.prob_head, self.score_head}
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nonlinearity = "linear" if m in output_heads else "relu"
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
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
        x_numeric : (batch, n_numeric)  — raw feature units; standardized here
        wh_ids    : (batch,)             — warehouse index ∈ [0, n_warehouses)
        sku_ids   : (batch,)             — sku index ∈ [0, n_skus)

        Returns
        -------
        logit_prob : (batch, 1) — raw logit for BCEWithLogitsLoss
        score      : (batch, 1) — sigmoid risk score for MSELoss
        """
        x = torch.cat(
            [self.standardize(x_numeric), self.wh_embedding(wh_ids), self.sku_embedding(sku_ids)],
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
