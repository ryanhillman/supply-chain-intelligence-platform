"""
Trainer — training loop with early stopping, LR scheduling, and checkpointing.

Owns:
  ShortageDataset — PyTorch Dataset that applies vocab mapping and tensor conversion.
  Trainer         — orchestrates training, validation, and checkpoint saving.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml_training.datasets.feature_engineer import (
    BASELINE_COL,
    NUMERIC_FEATURE_COLS,
    RISK_SCORE_COL,
    TARGET_COL,
)
from ml_training.evaluation.metrics import compute_metrics
from ml_training.training.config import TrainingConfig
from ml_training.training.loss import build_loss
from ml_training.training.model import ShortagePredictor

logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ShortageDataset(Dataset):
    """
    Wraps a Parquet-split DataFrame for DataLoader consumption.

    Numeric features are expected to already be scaled (StandardScaler applied
    in DatasetBuilder). Categorical IDs are mapped to integer indices via the
    vocabularies stored in feature_metadata.json.
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        wh_vocab:  dict[str, int],
        sku_vocab: dict[str, int],
    ) -> None:
        self.x_num    = torch.tensor(df[NUMERIC_FEATURE_COLS].values, dtype=torch.float32)
        self.wh_ids   = torch.tensor(
            [wh_vocab[w] for w in df["warehouse_id"]], dtype=torch.long
        )
        self.sku_ids  = torch.tensor(
            [sku_vocab[s] for s in df["sku_id"]], dtype=torch.long
        )
        self.y_binary = torch.tensor(
            df[TARGET_COL].values.astype(float), dtype=torch.float32
        ).unsqueeze(1)
        self.y_risk   = torch.tensor(
            df[RISK_SCORE_COL].values.astype(float), dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(self, idx: int):
        return (
            self.x_num[idx],
            self.wh_ids[idx],
            self.sku_ids[idx],
            self.y_binary[idx],
            self.y_risk[idx],
        )


# ── Training history ──────────────────────────────────────────────────────────

@dataclass
class TrainingHistory:
    train_loss:  list[float] = field(default_factory=list)
    val_loss:    list[float] = field(default_factory=list)
    val_auc_roc: list[float] = field(default_factory=list)
    val_auprc:   list[float] = field(default_factory=list)
    val_brier:   list[float] = field(default_factory=list)
    best_epoch:  int         = 0
    best_val_auprc: float    = -1.0


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Parameters
    ----------
    config:
        TrainingConfig with all hyperparameters.
    dataset_dir:
        Path to the directory containing train.parquet, val.parquet,
        feature_metadata.json, and preprocessor.pkl.
    output_dir:
        Directory where model.pt and training_metrics.json are written.
    """

    def __init__(
        self,
        config:      TrainingConfig,
        dataset_dir: str | Path,
        output_dir:  str | Path,
    ) -> None:
        self.config      = config
        self.dataset_dir = Path(dataset_dir)
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _seed_everything(config.seed)

    def train(self) -> TrainingHistory:
        """Run the full training loop. Returns TrainingHistory."""
        metadata = _load_metadata(self.dataset_dir / "feature_metadata.json")
        train_df = pd.read_parquet(self.dataset_dir / "train.parquet")
        val_df   = pd.read_parquet(self.dataset_dir / "val.parquet")

        wh_vocab  = metadata["warehouse_vocab"]
        sku_vocab = metadata["sku_vocab"]

        train_ds = ShortageDataset(train_df, wh_vocab, sku_vocab)
        val_ds   = ShortageDataset(val_df,   wh_vocab, sku_vocab)

        # drop_last=True: BatchNorm1d raises on a training batch of size 1,
        # which happens whenever len(train) % batch_size == 1.
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,  drop_last=True
        )
        val_loader = DataLoader(
            val_ds,   batch_size=self.config.batch_size, shuffle=False, drop_last=False
        )

        model = ShortagePredictor(
            n_numeric    = metadata["n_numeric_features"],
            n_warehouses = metadata["n_warehouses"],
            n_skus       = metadata["n_skus"],
            config       = self.config,
        )

        pos_weight = float(metadata.get("pos_weight", 1.0))
        criterion  = build_loss(self.config, pos_weight)
        optimizer  = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        scheduler  = _build_scheduler(optimizer, self.config)

        history    = TrainingHistory()
        best_path  = self.output_dir / "model_best.pt"
        patience_counter = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._train_epoch(model, criterion, optimizer, train_loader)
            val_metrics   = self._eval_epoch(model,  criterion, val_loader)

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["auprc"])
            else:
                scheduler.step()

            history.train_loss.append(train_metrics["loss"])
            history.val_loss.append(val_metrics["loss"])
            history.val_auc_roc.append(val_metrics["auc_roc"])
            history.val_auprc.append(val_metrics["auprc"])
            history.val_brier.append(val_metrics["brier"])

            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f "
                "| val_auprc=%.4f | val_auc_roc=%.4f | val_brier=%.4f",
                epoch,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["auprc"],  val_metrics["auc_roc"], val_metrics["brier"],
            )

            if val_metrics["auprc"] > history.best_val_auprc:
                history.best_val_auprc = val_metrics["auprc"]
                history.best_epoch     = epoch
                patience_counter       = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "config":      self.config,
                    "val_metrics": val_metrics,
                }, best_path)
                logger.info("  [best] AUPRC=%.4f — checkpoint saved", history.best_val_auprc)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping at epoch %d (no val AUPRC improvement for %d epochs)",
                        epoch, self.config.patience,
                    )
                    break

        # Promote best checkpoint to model.pt
        # Use replace() instead of rename() — on Windows, rename() fails if the
        # target already exists (from a previous training run).
        final_path = self.output_dir / "model.pt"
        if best_path.exists():
            best_path.replace(final_path)

        # Save training metrics
        _save_training_metrics(history, self.output_dir / "training_metrics.json")
        logger.info(
            "Training complete — best epoch=%d, best val AUPRC=%.4f",
            history.best_epoch, history.best_val_auprc,
        )
        return history

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model:     nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader:    DataLoader,
    ) -> dict[str, float]:
        model.train()
        total_loss = 0.0

        for x_num, wh_ids, sku_ids, y_binary, y_risk in loader:
            optimizer.zero_grad()
            logit_prob, pred_score = model(x_num, wh_ids, sku_ids)
            loss, _, _ = criterion(logit_prob, pred_score, y_binary, y_risk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / max(len(loader), 1)}

    @torch.no_grad()
    def _eval_epoch(
        self,
        model:     nn.Module,
        criterion: nn.Module,
        loader:    DataLoader,
    ) -> dict[str, float]:
        model.eval()
        total_loss  = 0.0
        all_probs:  list[float] = []
        all_labels: list[float] = []

        for x_num, wh_ids, sku_ids, y_binary, y_risk in loader:
            logit_prob, pred_score = model(x_num, wh_ids, sku_ids)
            loss, _, _ = criterion(logit_prob, pred_score, y_binary, y_risk)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logit_prob).squeeze(1).tolist())
            all_labels.extend(y_binary.squeeze(1).tolist())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        metrics = {"loss": total_loss / max(len(loader), 1)}
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            metrics.update(compute_metrics(y_true, y_prob))
        else:
            # AUC-ROC and AUPRC are undefined with only one class present,
            # but Brier score (mean squared error of the probability against
            # the single-class outcome) is well-defined regardless — compute
            # it rather than hardcoding a placeholder.
            metrics.update({
                "auc_roc": 0.5,
                "auprc":   float(y_true.mean()),
                "brier":   float(np.mean((y_prob - y_true) ** 2)),
            })

        return metrics


# ── Utilities ─────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config:    TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.lr_factor,
            patience=config.lr_patience,
        )
    # One cosine period over the whole epoch budget. A shorter T_max (the old
    # default of 20 against max_epochs=100) restarts the schedule every T_max
    # epochs, cycling the LR back up to its initial value.
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max or config.max_epochs, eta_min=1e-6,
    )


def _save_training_metrics(history: TrainingHistory, path: Path) -> None:
    out = {
        "best_epoch":     history.best_epoch,
        "best_val_auprc": history.best_val_auprc,
        "final_val_auc_roc": history.val_auc_roc[-1] if history.val_auc_roc else None,
        "final_val_brier":   history.val_brier[-1]   if history.val_brier   else None,
        "n_epochs_trained":  len(history.train_loss),
        "train_loss":     history.train_loss,
        "val_auprc":      history.val_auprc,
        "val_auc_roc":    history.val_auc_roc,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
