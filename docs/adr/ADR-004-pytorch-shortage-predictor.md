# ADR-004 — PyTorch Shortage Predictor

**Status:** Accepted  
**Date:** 2026-06-20  
**Phase:** 4  
**Author:** Supply Chain Intelligence Platform

---

## Context

Phase 3 completed the Silver feature store: 29 features across 6 column families
per (warehouse_id, sku_id) entity, written to GCS Parquet and Bigtable.

Two Silver columns currently contain null placeholders:

| Column | Family | Phase 4 target |
|---|---|---|
| `risk__shortage_prob_7d` | risk | P(stockout ≤ 7d) ∈ [0, 1] |
| `risk__stockout_risk_score` | risk | Continuous risk score ∈ [0, 1] |

Phase 4 must define and train a model to populate these columns and provide a
defensible business justification that the model improves on a rule-based baseline.

---

## Decision 1 — Framework: PyTorch

**Chosen:** PyTorch (`torch >= 2.2.0`, CPU-only training image)

**Rejected:**
- XGBoost / LightGBM — no native ONNX embedding support; harder to extend to
  sequence models in Phase 6; excluded by project constraint
- scikit-learn models as final model — excluded by project constraint
  (sklearn may be used for preprocessing utilities only)

**Rationale:**
1. **Phase 5 readiness**: Vertex AI Model Registry accepts ONNX. `torch.onnx.export`
   produces a compatible graph with dynamic batch axis. LightGBM requires a separate
   ONNX converter with known edge-case failures on categorical features.
2. **Entity embeddings**: PyTorch `nn.Embedding` layers learn latent
   warehouse and SKU representations from data. A gradient-boosted tree would
   require one-hot encoding (12 binary columns for 6 warehouses + 20 SKUs),
   which loses the geometric structure that embeddings capture.
3. **Extensibility**: The shallow MLP trunk can be replaced with a Transformer
   or LSTM in Phase 6 if multi-step Silver snapshots accumulate, without
   changing the serving infrastructure.

---

## Decision 2 — Target Definition

**Primary target:** `stockout_label_7d` (binary, 0 or 1)  
Definition: did `quantity_after` reach 0 at any point in (T, T+7d] in the
Bronze `inventory.events` event stream for the same (warehouse_id, sku_id)?

**Secondary target:** `continuous_risk_score` (float ∈ [0, 1])  
Definition: P(days-to-stockout ≤ 7) under a Normal approximation, where
`days-to-stockout ~ N(μ, σ²)`:
```
μ = current_qty / avg_daily_sales_7d
σ = current_qty × stddev_daily_7d / avg_daily_sales_7d²
```
This is the regression target for the `risk__stockout_risk_score` output head.
It provides a smooth training signal and handles demand uncertainty explicitly.

**Prediction horizon:** 7 calendar days.

---

## Decision 3 — Label Generation Strategy: Bronze Temporal Sampling

**Problem with Silver-only labels:**
`inventory__days_of_supply = current_qty / avg_daily_sales_7d` already exists
as a Silver feature. Labeling each Silver row by `(days_of_supply < 7)`
produces a label that is a deterministic threshold on a feature the model
already sees. The model would trivially recover the baseline rule with near-zero
training error but learn nothing generalizable about demand uncertainty.

**Chosen approach:** Bronze temporal sampling.
For each (warehouse_id, sku_id), sample one row per calendar day from the
Bronze `inventory.events` stream. For each sample at date T:
- **Features**: computed from events at or before T (no future data)
- **Label**: derived from events in (T, T+7d] — strictly after the feature window

This eliminates all feature-to-label leakage. The model must predict the future
from the present, which is the actual operational prediction task.

**Estimated dataset size (at ADR authoring time):** ~3,000–7,000 labeled
examples (120 pairs × 30–60 available sample dates after applying 30-day
lookback and 7-day lookahead buffers), based on real accumulating Bronze
data.

**Update:** the implemented training pipeline instead uses a synthetic data
generator (`ml_training/datasets/synthetic_generator.py`) that simulates 365
days across all 120 (warehouse, SKU) pairs, producing ~43,800 labeled
examples — see the README's Training Data section. The leakage-control
principle above (features from events at or before T, label from events in
(T, T+7d]) applies identically regardless of data source; only the source
and resulting scale changed. The "Why not TabNet" rationale below was
written against the original ~3,000–7,000 estimate and was not
re-evaluated at the larger scale — see that section's note.

---

## Decision 4 — Feature Selection and Leakage Controls

**Included features (21 numeric + 2 categorical):**

| Group | Features |
|---|---|
| Inventory state | current_qty, reorder_point, safety_stock |
| Velocity | sales_7d, sales_30d, avg_daily_sales_{7d,30d}, velocity_trend, stddev_daily_7d |
| Supplier | lead_time_days, lead_time_stddev, supplier_reliability |
| Pipeline | active_po_qty, in_transit_qty |
| Demand | demand_signal_count_24h |
| Derived | days_of_supply, effective_supply, buffer_ratio, demand_cv, supply_coverage_days, replenishment_gap |
| Categorical | warehouse_id → Embedding(6, 3), sku_id → Embedding(20, 5) |

**Excluded (leakage or target):**

| Column | Reason |
|---|---|
| `risk__reorder_urgency` | Derived from the same threshold logic as the label |
| `risk__stockout_risk_score` | Target output |
| `risk__shortage_prob_7d` | Target output |
| `demand__demand_forecast_{7d,30d,multiplier}` | Reserved null columns |
| All `lineage__*` | Pipeline metadata, no predictive value |

**Time-based split (no random leakage):**
```
Train : earliest dates → 70th percentile date
Val   : 70th → 85th percentile date
Test  : 85th percentile date → latest date
```
Future events are never visible in features. The test split is held out entirely
until final evaluation — it is not used for hyperparameter tuning or early stopping.

---

## Decision 5 — Model Architecture: Shallow MLP with Entity Embeddings

```
warehouse_id  → Embedding(6,  dim=3)  ┐
sku_id        → Embedding(20, dim=5)  ┤ concatenated → [29]
numeric (21)  → StandardScaler        ┘
                      ↓
    Linear(29→64) + BatchNorm1d + ReLU + Dropout(0.30)
                      ↓
    Linear(64→32) + BatchNorm1d + ReLU + Dropout(0.20)
                      ↓
    Linear(32→16) + BatchNorm1d + ReLU
                      ↓
    prob_head  → logit  → BCEWithLogitsLoss  → shortage_prob_7d
    score_head → sigmoid → MSELoss           → stockout_risk_score
```

**Why not TabNet:** TabNet's sparse attention requires ≥10,000 training examples
to generalize; on 3,000–7,000 rows a regularized MLP matches TabNet in practice.
**Note:** this was written against the ~3,000–7,000-row estimate above. The
implemented pipeline's synthetic generator produces ~43,800 rows, which
exceeds TabNet's stated threshold — this specific "insufficient data"
argument no longer holds as stated, and the choice was not re-benchmarked
against TabNet at the larger scale. The simpler MLP remains a reasonable
default given the small, fixed, well-understood feature set (21 numeric + 2
embeddings), but that's a different justification than the one originally
given here.

**Why not LSTM/Transformer:** Silver features are point-in-time snapshots, not
sequences. Temporal structure is already encoded in the 7d/30d velocity windows.
A sequence model would require aligned multi-step history per entity — a Phase 6
concern once more Silver epochs have accumulated.

**Why entity embeddings:** Warehouse and SKU IDs are not ordinal. One-hot
encoding would add 26 binary columns that lose geometric structure. Embeddings
(dim 3 and 5 respectively) learn latent representations — e.g., the embedding
space may cluster warehouses with similar demand patterns — without adding
unnecessary dimensionality for a small dataset.

---

## Decision 6 — Evaluation Metrics and Success Criterion

**Phase 4 success criterion:** The trained PyTorch model must outperform the
closed-form baseline on ALL three metrics on the held-out test split.

**Baseline:**
```python
continuous_risk_score = Φ((7 − μ) / σ)     # P(stockout ≤ 7d), Normal approximation
μ = current_qty / avg_daily_sales_7d       # mean days to zero
σ = current_qty · stddev_daily_7d / avg_daily_sales_7d²
```
This is the strongest thing the business could compute without a model: a
calibrated probability from three features the model also sees. If the network
cannot beat it, it is not adding value.

The simpler 0/1 rule `baseline_pred = float(days_of_supply < 7)` is still
computed and reported for context, but it is no longer the bar. A binary
predictor has structurally low AUC-ROC and AUPRC (on the current synthetic test
split: 0.84 / 0.45, versus 0.87 / 0.59 for the formula and 0.92 / 0.65 for
simply ranking by `days_of_supply`), so outperforming it proved very little.

| Metric | Direction | Rationale |
|---|---|---|
| AUPRC (Average Precision) | Higher is better | Primary metric; most informative under class imbalance |
| AUC-ROC | Higher is better | Overall discriminative ability |
| Brier Score | Lower is better | Probability calibration quality |

No fixed numeric thresholds are hardcoded. The comparison to the baseline
provides a naturally calibrated, self-updating target: if the data changes
in production, the baseline automatically adjusts.

**Per-warehouse breakdown** is computed as a secondary diagnostic. If the model
underperforms the baseline for a specific warehouse, that signals a data quality
or distribution issue for that location.

---

## Decision 7 — Model Persistence

**Artifact structure** written by `scripts/train.py`:
```
ml_training/artifacts/
  model.pt              — torch.save checkpoint: {epoch, model_state, config, val_metrics}
  model.onnx            — ONNX opset 17, dynamic batch axis, validated with onnxruntime
  feature_metadata.json — vocab mappings, feature list, pos_weight, split boundaries
  training_metrics.json — per-epoch train/val AUPRC, AUC-ROC, loss curves
  evaluation_test.json  — full model vs baseline comparison on test split
```

**Local:** `ml_training/artifacts/` (gitignored via root `.gitignore`: `ml_training/artifacts/*.pt` etc.)

**GCS (optional upload):**
```
gs://{bucket}/models/shortage-predictor/v{YYYYMMDD_HHMMSS}/
```
Timestamped versioning enables champion/challenger comparison in Phase 5.

**StandardScaler** (`preprocessor.pkl`) is written to `ml_training/artifacts/dataset/`
(the dataset directory, not the model directory) because it is a dataset-level
artifact. The `ShortagePredictor` inference class accepts a `dataset_dir` parameter
to locate it.

---

## Decision 8 — Retraining Strategy

**Trigger:** Event-driven, not calendar-driven.
- When the Silver batch job produces a new GCS epoch, a retraining run may be
  scheduled (Phase 5 orchestration to define the exact trigger).
- Minimum retraining interval: weekly (to accumulate meaningful new Bronze data).

**Champion/challenger:**
- A new model version is only promoted if it outperforms the current champion
  on the held-out test split (using the same three-metric criterion).
- The test split cutoff date advances with each retraining (rolling window).

**Feature drift detection (Phase 5):**
- PSI (Population Stability Index) on key features (days_of_supply, sales_7d,
  supplier_reliability) monitored weekly.
- PSI > 0.2 triggers an alert and a retraining recommendation.

---

## Consequences

**Positive:**
- Phase 4 populates the two null Silver placeholders with meaningful predictions.
- The model provides a defensible improvement over the heuristic baseline,
  quantified by AUPRC / AUC-ROC / Brier on held-out data.
- ONNX export provides a direct path to Vertex AI serving in Phase 5.

**Accepted trade-offs:**
- The original Bronze-only sampling design targeted a small training set
  (3,000–7,000 examples); the implemented pipeline's synthetic generator
  produces ~43,800 (see the Update note under Decision 3). Either way, the
  model is validated against the baseline rather than against an absolute
  threshold, so the success criterion doesn't need to be re-tuned if the
  dataset size changes again.
- The Bronze temporal sampler recomputes rolling features in Python (not Spark).
  For the current Bronze dataset size (~53,000 events), Python + pandas is
  adequate. If Bronze grows to millions of events, the sampler should be
  reimplemented as a Spark batch job.
- `continuous_risk_score` uses a Normal approximation for its label; this
  approximation degrades when daily demand has heavy tails. The MSE loss weight
  (0.3) keeps this secondary target from dominating training.
