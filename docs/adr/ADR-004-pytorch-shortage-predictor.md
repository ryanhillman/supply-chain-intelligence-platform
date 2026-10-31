# ADR-004: PyTorch Shortage Predictor

**Status:** Accepted, June 2026. LightGBM comparison added October 2026.

## Context

The model predicts, for each (warehouse, SKU) pair, the probability that stock runs out within the next 7 days. Inputs are 21 numeric features plus the warehouse and SKU identities. Training data comes from a simulator covering 365 days and 120 pairs, about 43,800 labeled rows.

## Decision 1 — PyTorch as a learning objective

PyTorch was selected to gain experience training a network end to end: embeddings, a training loop, early stopping, and ONNX export to a serving container. The problem does not require a neural network.

A same-split benchmark confirms this. Synthetic test split, 6,600 rows, 17.3% positive:

| Model | AUC-ROC | AUPRC | Brier |
|---|---|---|---|
| Rule: `days_of_supply < 7` | 0.842 | 0.446 | 0.184 |
| Normal-approximation formula | 0.871 | 0.593 | 0.166 |
| Logistic regression | 0.928 | 0.688 | 0.080 |
| LightGBM (120 trees) | 0.936 | 0.740 | 0.075 |
| PyTorch MLP, weighted BCE (original default) | 0.933 | 0.719 | 0.117 |
| PyTorch MLP, unweighted BCE (current default) | 0.932 | 0.732 | 0.076 |

LightGBM slightly exceeds the MLP on all three metrics and needs no standardization layer or embeddings. It is the recommended model for a production version of this system.

The deployment's cost comes from the managed infrastructure (the Vertex AI endpoint and the Bigtable node), not from the model. Either model can be scored inside the batch job; promotion rescoring already scores the MLP in-process with onnxruntime.

The MLP is retained as the learning artifact and as the comparison row that documents the choice as measured rather than assumed. The original version of this record rejected LightGBM on ONNX-export and embedding grounds. Both points are technically valid, but they were not the basis of the decision, and the benchmark supersedes them.

## Decision 2 — Target: stockout within 7 days

The label `stockout_label_7d` is 1 if `quantity_after` reaches 0 at any point in (T, T+7d] for that pair.

`stockout_risk_score` is not a model output. It is the closed-form Normal-approximation probability computed from current stock, average daily sales and their standard deviation. It is the baseline (Decision 6), and the scorer publishes it next to the model's probability. An earlier version trained a second head to regress this formula, which only approximated a function of three of the model's own inputs; that head was removed.

## Decision 3 — Labels from the event history, not from Silver

Labeling a Silver row by `days_of_supply < 7` makes the label a threshold on a feature the model already sees, so the model would learn the rule back. Instead, each sample at date T takes its features from events at or before T and its label from events in (T, T+7d]. The model predicts the future from the present, which is the operational task.

The original plan sampled real Bronze history, which required weeks of accumulated events. The training set comes from the simulator instead. The leakage rule is identical for either source.

## Decision 4 — Features and a time-based split

Numeric features (21): current stock and reorder settings, 7- and 30-day sales velocity, supplier lead time and reliability, open purchase orders, in-transit quantity, the 24-hour demand-signal count, and six derived ratios (days of supply, buffer ratio, and others). Warehouse and SKU are embedded (dimensions 3 and 5).

Excluded: `reorder_urgency`, built from the same threshold logic as the label; the two `risk` outputs; the reserved forecast columns; lineage metadata.

The split is by date: the first 70% of dates train, the next 15% validate, the last 15% test, with a 7-day gap between splits so no label window overlaps the next split's features. The test split is not used for tuning or early stopping.

## Decision 5 — A small MLP with entity embeddings

```
warehouse_id → Embedding(6, 3)  ┐
sku_id       → Embedding(20, 5) ┤ concat → 29
numeric (21) → Standardize      ┘
  → Linear(29→64) + BatchNorm + ReLU + Dropout(0.3)
  → Linear(64→32) + BatchNorm + ReLU + Dropout(0.2)
  → Linear(32→16) + BatchNorm + ReLU
  → logit → shortage_prob_7d
```

Standardization is a fixed layer inside the model, so the exported graph takes raw features (ADR-005 Decision 3). Training uses AdamW, unweighted BCE, cosine LR decay over the full run, and early stopping on validation AUPRC. Class weighting did not improve ranking and degraded calibration (Brier 0.117 vs 0.076), so it is off by default.

## Decision 6 — Beat the formula, not the rule

The model must beat the Normal-approximation formula on all three test metrics: AUPRC (primary, since positives are a minority), AUC-ROC, and Brier score for calibration.

```
continuous_risk_score = Φ((7 − μ) / σ)
μ = current_qty / avg_daily_sales_7d
σ = current_qty · stddev_daily_7d / avg_daily_sales_7d²
```

The formula is the strongest prediction available without a model, computed from three features the model also sees. The initial version compared against the 0/1 rule `days_of_supply < 7`. A binary predictor has structurally low AUC and AUPRC (0.84 / 0.45 on the test split, against 0.87 / 0.59 for the formula and 0.92 / 0.65 for ranking by `days_of_supply`), so beating it demonstrated little. The rule is still reported for context.

Evaluation also reports in-stock rows separately. Rows already at zero stock are easy positives and inflate the headline metrics.

## Decision 7 — Training artifacts

`ml_training/scripts/train.py` writes to `ml_training/artifacts/`:

- `model.pt` and `model.onnx` (opset 17, dynamic batch axis, checked against PyTorch with onnxruntime)
- `feature_metadata.json`: feature order, vocabularies, standardization mean and scale
- `evaluation_test.json`: model, formula and rule metrics, plus a sha256 of the test split
- `training_metrics.json`: per-epoch curves
- `drift_reference.json`: training-distribution bins for the drift monitor (ADR-005 Decision 8)

`model_serving/scripts/register.py` uploads these to `gs://{bucket}/models/shortage-predictor/v{timestamp}/` for Vertex AI.

## Decision 8 — Retraining

Retraining is manual. The drift monitor recommends it (ADR-005 Decision 7), and a new model reaches traffic only if it beats the current one on the same test split (ADR-005 Decision 6). No scheduler triggers a retrain; at this scale none is needed.

## At scale

Replacing the MLP with LightGBM would keep the data path (Kafka, Spark, Bigtable, Pub/Sub, the feature client). Training, evaluation, ONNX export, the serving container's model loader and promotion rescoring are written for this network's graph (three inputs, one logit output) and would change.

A neural model would be worth re-evaluating against LightGBM if the problem changed shape:

- **Sequence input:** raw event history instead of 7- and 30-day window features.
- **Large catalogs with cold start:** embeddings sharing signal across SKUs, including new ones.
- **Multi-horizon or multi-task output:** several horizons or demand quantiles from one network.
- **Unstructured input:** product text, images, supplier documents.

## Consequences

- The model is measured against the strongest simple baseline, so a win is meaningful.
- The benchmark shows the learning objective added complexity, not accuracy.
- The Normal approximation behind the baseline and `stockout_risk_score` degrades when daily demand has heavy tails.
- The sampler computes rolling features in pandas. That is adequate at this data size; at millions of events it would move to Spark.
