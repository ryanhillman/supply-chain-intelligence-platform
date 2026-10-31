"""
Drift reference — PSI bins for the monitored features.

Computed from the raw training split when the model is trained and written
next to model.pt as drift_reference.json. The drift monitor reads this file,
so its reference distribution travels with the model artifacts rather than
with whatever train.parquet happens to be on the operator's disk — the same
provenance problem the old preprocessor.pkl had.

Bins are the training rows' deciles, deduplicated (clustered features such as
supplier_reliability collapse several decile edges onto one value), and
`expected` is each bin's actual training fraction rather than an assumed
1/10. Only the finite interior edges are stored; the outer edges are ±inf at
load time, so every live value lands in some bin.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# (display_name, flat feature name). The flat name must be in
# NUMERIC_FEATURE_COLS. days_of_supply is derived by engineer_features() on
# both the training and the live side, so both are in the same units.
MONITORED_FEATURES: list[tuple[str, str]] = [
    ("inventory__days_of_supply",      "days_of_supply"),
    ("velocity__sales_7d",             "sales_7d"),
    ("velocity__avg_daily_sales_7d",   "avg_daily_sales_7d"),
    ("supplier__supplier_reliability", "supplier_reliability"),
]

N_BINS: int = 10


def build_drift_reference(train_df: pd.DataFrame, source: str = "train.parquet") -> dict:
    """
    Decile bins and expected fractions for every monitored feature.

    Parameters
    ----------
    train_df:
        The training split in raw feature units (as DatasetBuilder writes it).
    source:
        Recorded in the file for provenance.
    """
    quantile_points = np.linspace(0.0, 1.0, N_BINS + 1)
    features: dict[str, dict] = {}

    for display_name, name in MONITORED_FEATURES:
        if name not in train_df.columns:
            raise ValueError(f"Training split has no column {name!r} for the drift reference")
        values = train_df[name].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"No training values for monitored feature {name!r}")

        edges    = np.unique(np.quantile(values, quantile_points))
        interior = edges[1:-1]                      # finite edges between the extremes
        counts, _ = np.histogram(values, bins=_full_edges(interior))

        features[name] = {
            "display_name":   display_name,
            "interior_edges": [float(e) for e in interior],
            "expected":       (counts / len(values)).tolist(),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source":     source,
        "n_rows":     int(len(train_df)),
        "n_bins":     N_BINS,
        "features":   features,
    }


def write_drift_reference(reference: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return path


def load_drift_reference(path: str | Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """
    Returns ({feature: (bin_edges incl. ±inf, expected)}, info) where info
    carries created_at, source, n_rows and n_bins for the monitor's report.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bins = {
        name: (_full_edges(np.asarray(spec["interior_edges"], dtype=float)),
               np.asarray(spec["expected"], dtype=float))
        for name, spec in data["features"].items()
    }
    info = {k: data.get(k) for k in ("created_at", "source", "n_rows", "n_bins")}
    return bins, info


def _full_edges(interior: np.ndarray) -> np.ndarray:
    return np.concatenate(([-np.inf], np.asarray(interior, dtype=float), [np.inf]))
