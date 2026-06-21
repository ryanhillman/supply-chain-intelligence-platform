"""
ShortagePredictor (inference) — loads trained model artifacts and runs
batch inference on a feature DataFrame.

Populates the two Phase 4 target columns:
  risk__shortage_prob_7d    — P(stockout ≤ 7d) ∈ [0, 1]
  risk__stockout_risk_score — continuous risk score ∈ [0, 1]

Designed to be called from scripts/predict_silver.py after reading Silver
features via datasets/silver_snapshot_loader.py.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase4.datasets.feature_engineer import NUMERIC_FEATURE_COLS
from phase4.training.config import TrainingConfig
from phase4.training.model import ShortagePredictor as _ShortageModel

logger = logging.getLogger(__name__)

# Column names written to the output DataFrame
OUT_PROB_COL:  str = "risk__shortage_prob_7d"
OUT_SCORE_COL: str = "risk__stockout_risk_score"


class ShortagePredictor:
    """
    Loads model artifacts from a directory and provides a predict() method.

    Artifact directory must contain:
      model.pt              — torch checkpoint with "model_state" key
      feature_metadata.json — vocab mappings and feature list

    The StandardScaler is loaded from the dataset_dir (not the artifact dir)
    because it is a dataset-level artifact shared across train/val/test.
    If dataset_dir is not supplied, the predictor expects pre-scaled features.

    Parameters
    ----------
    artifacts_dir:
        Directory produced by scripts/train.py.
    dataset_dir:
        Directory containing preprocessor.pkl (from DatasetBuilder).
        If None, features are assumed to be already scaled.
    """

    def __init__(
        self,
        artifacts_dir: str | Path,
        dataset_dir:   str | Path | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._metadata     = self._load_json(self.artifacts_dir / "feature_metadata.json")

        self._wh_vocab  : dict[str, int] = self._metadata["warehouse_vocab"]
        self._sku_vocab : dict[str, int] = self._metadata["sku_vocab"]
        self._config                     = TrainingConfig()

        self._model = _ShortageModel(
            n_numeric    = self._metadata["n_numeric_features"],
            n_warehouses = self._metadata["n_warehouses"],
            n_skus       = self._metadata["n_skus"],
            config       = self._config,
        )
        ckpt = torch.load(
            self.artifacts_dir / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        logger.info(
            "Model loaded from %s (trained epoch=%s, val_auprc=%s)",
            self.artifacts_dir,
            ckpt.get("epoch", "?"),
            ckpt.get("val_metrics", {}).get("auprc", "?"),
        )

        self._scaler = None
        if dataset_dir is not None:
            scaler_path = Path(dataset_dir) / "preprocessor.pkl"
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("StandardScaler loaded from %s", scaler_path)
            else:
                logger.warning("preprocessor.pkl not found at %s — features assumed pre-scaled.", scaler_path)

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a feature DataFrame.

        Parameters
        ----------
        feature_df:
            Must contain all NUMERIC_FEATURE_COLS and categorical columns
            warehouse_id, sku_id. Typically the output of
            silver_snapshot_loader.load_silver_features().

        Returns
        -------
        pd.DataFrame
            Input DataFrame with two new columns appended:
            risk__shortage_prob_7d and risk__stockout_risk_score.
        """
        df = feature_df.copy()

        # Scale numeric features if scaler is available
        x_num_raw = df[NUMERIC_FEATURE_COLS].values.astype(np.float32)
        if self._scaler is not None:
            x_num_scaled = self._scaler.transform(x_num_raw).astype(np.float32)
        else:
            x_num_scaled = x_num_raw

        x_num   = torch.tensor(x_num_scaled, dtype=torch.float32)
        wh_ids  = torch.tensor(
            [self._wh_vocab.get(w, 0) for w in df["warehouse_id"]], dtype=torch.long
        )
        sku_ids = torch.tensor(
            [self._sku_vocab.get(s, 0) for s in df["sku_id"]], dtype=torch.long
        )

        with torch.no_grad():
            prob, score = self._model.predict(x_num, wh_ids, sku_ids)

        df[OUT_PROB_COL]  = prob.squeeze(1).numpy()
        df[OUT_SCORE_COL] = score.squeeze(1).numpy()

        logger.info(
            "Predictions generated: %d rows | prob mean=%.3f | score mean=%.3f",
            len(df),
            float(df[OUT_PROB_COL].mean()),
            float(df[OUT_SCORE_COL].mean()),
        )
        return df

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)
