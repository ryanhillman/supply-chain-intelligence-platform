# ADR-005: Vertex AI Serving and Prediction Storage

**Status:** Accepted, June 2026

## Context

Training produces `model.onnx`. Serving needs a runtime for it, a store for its predictions, and a way to detect when those predictions stop being trustworthy. The batch scorer reads 120 feature rows from Bigtable, scores them, and stores one prediction per (warehouse, SKU) pair.

Vertex AI (Model Registry and an online endpoint) was selected as a learning objective: model registration, deployment, traffic splitting and promotion on a managed platform. The workload does not require it: a model this small scoring 120 rows can run inside the scorer process. An endpoint bills by the node-hour while a model is deployed, regardless of traffic, so models are undeployed between demo sessions (`deploy.py undeploy`).

## Decision 1 — Predictions are not written back to Silver

Silver is recomputed from Bronze and treated as append-only. Predictions change with every retrain, so writing them into Silver would require overwriting partitions, mix data of two origins, and break the guarantee that Silver can be rebuilt from Bronze at any time. The two `risk` prediction columns remain null in Silver.

## Decision 2 — Predictions live in a `predictions` column family in Bigtable

Predictions are stored in the existing `inventory_features` table, under the same `{warehouse_id}#{sku_id}` row key, in a dedicated family:

- `shortage_prob_7d`: the model's probability
- `stockout_risk_score`: the closed-form baseline (ADR-004 Decision 2)
- `model_version`, e.g. `v20260620_180918`
- `prediction_timestamp`, UTC

The family keeps the last 3 versions: the current prediction plus two earlier rounds, enough to compare a challenger's output with the champion's. A separate table would duplicate IAM and GC configuration with no benefit, and a shared row lets one scan return features and the latest prediction together.

## Decision 3 — Standardization lives inside the model graph

`ShortagePredictor` standardizes its own numeric inputs. The per-feature mean and scale are fit on the training split, stored in `feature_metadata.json`, and registered as fixed buffers, so they ship inside `model.pt` and `model.onnx`. The endpoint takes raw feature values and the scorer sends them unchanged.

The initial design kept the scaler in a pickled `preprocessor.pkl` loaded by the scorer from the local dataset directory, on the premise that the scaler was a dataset artifact rather than part of the model. That premise was wrong. The weights are only meaningful for inputs scaled with exactly those statistics, and nothing tied the pickle on disk to the model version the endpoint served. A local retrain without a redeploy would have scored every row with the wrong statistics. With the scaler inside the graph, the ONNX file is self-contained, the serving side no longer depends on scikit-learn, and the client's only preprocessing is mapping warehouse and SKU strings to vocab indices.

## Decision 4 — A small FastAPI container around ONNX Runtime

A custom container serves the model: a small FastAPI app that loads `model.onnx` with onnxruntime and implements Vertex's `/health` and `/predict` contract. The same image runs locally with `docker run`, which allows testing before deployment.

Inputs: `numeric_features` (float32, `[batch, 21]`, raw units), `warehouse_ids` and `sku_ids` (int64, `[batch]`). Output: `shortage_prob_7d`, the sigmoid of the graph's single logit. The container reads the feature width from the model and rejects a mismatched request with HTTP 422 instead of failing inside the graph.

## Decision 5 — Predictions go through Pub/Sub

The scorer publishes one JSON message per prediction to the `prediction-results` topic, and a separate writer drains the subscription into Bigtable. A message is acknowledged only after its Bigtable write succeeds, and a dead-letter topic receives anything that fails five delivery attempts.

For 120 rows this hop exceeds what the workload needs; the scorer could write the rows directly. It is retained to exercise the decoupled scorer/writer pattern, and it allows additional consumers, such as alerting, to subscribe without changes to the scorer.

```json
{
  "warehouse_id": "W-001",
  "sku_id": "SKU-005",
  "shortage_prob_7d": 0.73,
  "stockout_risk_score": 0.68,
  "model_version": "v20260620_180918",
  "prediction_timestamp": "2026-06-20T18:09:21Z"
}
```

## Decision 6 — A challenger must beat the champion on the same test split

A new model deploys at 0% traffic and is promoted to 100% only if `promote_model.py` finds it better than the current champion on all three metrics: AUPRC, AUC-ROC and Brier. The previous version stays deployed at 0% for rollback.

Both models must be scored on the same holdout. Each `evaluation_test.json` records a sha256 of the test split it was computed on. If the hashes match, the stored metrics are compared directly. After a retrain on new data they will not match, so `promote_model.py` downloads both models and scores them on the challenger's `test.parquet` with onnxruntime. The local file's hash must equal the challenger's recorded hash; otherwise promotion is refused rather than compared on the wrong data. The initial version compared each model's metrics from its own test split, which is meaningless when the splits differ.

---

## Decision 7 — Distribution Shift as a First-Class Architecture Concern

**Background — what Phase 4 inference revealed:**

Phase 4 training used temporal simulation: Bronze events from one real day (2026-
06-20, ~6 hours) were tiled 60× with ±10% quantity noise, producing 23 simulated
sample dates. This inflated `avg_daily_sales_7d` by 13×:

| Feature | Training distribution (simulated) | Real Silver value | Deviation |
|---|---|---|---|
| `avg_daily_sales_7d` | mean ≈ 2,785 | 207 | −9.5σ |
| `days_of_supply` | mean ≈ 0.15 | 3.45 | +46σ |

At inference, real Silver features were 46σ outside the training distribution. The
model's probability output saturated to `shortage_prob_7d = 0.0` and
`stockout_risk_score = 1.0` for all 120 entities — mathematically precise but
operationally meaningless.

**Why model deployment alone is insufficient:**

A model producing uniformly wrong predictions is worse than no model in one
specific way: it produces confidence without accuracy. Downstream systems
consuming `predictions:shortage_prob_7d` have no native signal that the values are
stale or distribution-shifted. Without monitoring:

- A retraining run fixes the distribution shift, but the window between deployments
  produces silent wrong predictions with no alert.
- A Silver schema change (new feature, renamed column) silently degrades inference
  quality with no observable error.
- The model's per-entity variance collapses to zero on distribution-shifted input
  (all predictions converge to the same saturation value), which is detectable via
  score variance but only if someone is looking.

**What distribution shift monitoring must detect:**

1. **Feature drift (PSI):** Population Stability Index > 0.2 on any of the four
   primary features (`days_of_supply`, `sales_7d`, `avg_daily_sales_7d`,
   `supplier_reliability`) signals that the live feature distribution has diverged
   from the training distribution beyond the acceptable threshold.

   PSI thresholds:
   - PSI < 0.1: stable, no action
   - 0.1 ≤ PSI < 0.2: monitor, flag in logs
   - PSI ≥ 0.2: retraining trigger — flagged in the drift report (see
     Implementation note below: this is advisory, not an automated block)

2. **Prediction score collapse:** If the standard deviation of `shortage_prob_7d`
   across the 120 entities falls below 0.01, all entities have converged to the
   same probability. This is the observable symptom of the Phase 4 saturation
   failure and should raise a monitoring alert regardless of PSI.

3. **Baseline divergence:** If the model's predictions agree with the rule-based
   `risk__reorder_urgency` heuristic for > 95% of entities, the ML model adds no
   signal over the heuristic and retraining is warranted.

**Retraining trigger conditions (any one is sufficient):**

| Trigger | Threshold | Action |
|---|---|---|
| Feature PSI | ≥ 0.2 on any primary feature | Flagged in drift report; retraining recommended |
| Prediction variance | std(shortage_prob_7d) < 0.01 | Flagged in drift report; retraining recommended |
| Baseline divergence | agreement with reorder_urgency > 95% | Flagged in drift report; retraining recommended |
| Real Bronze data | ≥ 30 new real days accumulated | Retrain with `--simulate-days 0` |
| Minimum interval | 7 days since last retrain | Throttle — do not retrain on noisy daily PSI spikes |

**Implementation note — this is advisory, not an automated gate:** all three
triggers above are implemented in `drift_monitor.py` and produce a
`requires_retraining` flag plus human-readable reasons in the printed report
and `drift_report.json`. Nothing in `score.py`, `batch_scorer.py`, or
`promote_model.py` reads that flag back — none of them actually block or
reject anything today. "Action" in the table means the report surfaces the
recommendation for a human to act on, not that the pipeline enforces it.
Wiring an automated gate (e.g. `promote_model.py` refusing to promote when
the latest `drift_report.json` has `requires_retraining: true`) remains
future work.

**Production training requirement:**

The Phase 4 model was trained on simulated data and is a pipeline validation
artifact, not a production model. Production training requires accumulating ≥ 38
real calendar days of Bronze events (`--simulate-days 0`), which eliminates the
13× velocity inflation and produces a realistic `days_of_supply` distribution.

Until real multi-day Bronze data is available, the monitoring layer's primary
function is detecting and reporting the known Phase 4 distribution shift so a
human can withhold promotion, rather than allowing wrong predictions to
propagate to consumers unnoticed. See the implementation note above — this
detection is not yet wired into an automated block.

---

## Decision 8 — How the drift monitor computes PSI

`drift_monitor.py` reads current feature values from Bigtable, because those are exactly the values the model received. If Silver and Bigtable diverge, Bigtable is the model's actual input distribution, and monitoring Silver would miss the problem.

The reference distribution is `drift_reference.json`, written next to `model.pt` at training time: decile bin edges from the raw training split, plus the fraction of training rows observed in each bin. It travels with the model, so the monitor always compares against that model's own training data.

```
PSI = Σ (p_actual − p_expected) × ln(p_actual / p_expected)
```

Empty bins use a floor of 0.0001 to avoid log(0).

Two earlier implementations were incorrect:

- Bins built from normal quantiles of the training mean and scale. The monitored features are skewed, and the training data compared against itself produced PSI of 6 or more: a false trigger on identical data.
- Raw live values binned against standardized training rows, because the training splits were stored standardized at the time. The same train-against-itself check produced PSI of 2 to 8.

The splits are now stored in raw units, so bins and live values share units by construction, and the monitor refuses to run without `drift_reference.json` rather than fall back to an approximation.

`supplier_reliability` is computed per warehouse, so a live snapshot holds six values and decile PSI over them trips on unchanged data (3.96 on the replayed world). It is checked instead per warehouse against that warehouse's training p1–p99, which the same file stores, and an out-of-range warehouse is reported as an informational warning rather than a retraining trigger: it usually reflects seasonality outside the training window.

With `--emit-cloud-monitoring`, the monitor also writes PSI, prediction spread, baseline agreement and the retraining flag as custom Cloud Monitoring metrics. No alert policy is defined in Terraform, so nothing alerts on them.

## Consequences

- Silver remains rebuildable from Bronze, and predictions keep three versions for comparison.
- The scorer and writer fail independently without losing predictions.
- A regressed model cannot reach traffic unnoticed, and a distribution-shifted one is flagged.
- PSI monitors inputs, not outcomes. Detecting a change in the real stockout rate requires labels, which arrive 7 days late.
- The managed components (endpoint, Pub/Sub, Model Registry) cost more and do more than 120 rows require. That is the accepted cost of the learning objective.
