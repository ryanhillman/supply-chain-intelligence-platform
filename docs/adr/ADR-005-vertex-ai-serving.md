# ADR-005 — Vertex AI Serving and Prediction Storage

**Status:** Accepted
**Date:** 2026-06-20
**Phase:** 5

---

## Context

Phase 4 produced a trained ONNX model (`model.onnx`) and a validated inference
pipeline. Phase 5 must operationalize that model — registering it in a versioned
registry, deploying it to a managed endpoint, and persisting its predictions in a
durable, low-latency store.

Three design questions arise:

1. **Where should predictions be written?** Phase 4 batch inference wrote a local
   `predictions.parquet`. That file is ephemeral and inaccessible to downstream
   systems.
2. **Should predictions coexist with features in Silver?** Silver is GCS Parquet
   produced by Spark batch jobs. Writing model output back into the same Parquet
   partition would make the feature layer mutable, violating its role as the
   append-only source of truth.
3. **Is model deployment alone sufficient?** Phase 4 inference revealed a severe
   training/inference distribution mismatch. A deployed model that silently
   produces wrong predictions is operationally worse than no model. Monitoring
   and retraining triggers are mandatory, not optional.

---

## Decision 1 — Silver Immutability: Predictions Are Not Written to Silver

**Chosen:** Silver GCS Parquet remains strictly read-only after Phase 3. Phase 5
does not modify, append to, or delete any Silver file.

**Rejected:** Writing `risk__shortage_prob_7d` and `risk__stockout_risk_score`
back into Silver Parquet.

**Rationale:**

| Concern | Impact of writing to Silver |
|---|---|
| Immutability | Predictions change on every model retrain. Silver partitions would need to be overwritten, destroying the lineage audit trail. |
| Schema coupling | Silver schema is owned by the Spark pipeline (Phase 3). ML predictions are owned by the serving layer (Phase 5). Coupling them in the same file creates a cross-cutting ownership conflict. |
| Partial write risk | A failed prediction run would leave Silver in a partially updated state. |
| Replayability | Silver can be recomputed from Bronze at any time. That property is only safe if Silver contains no data with a different origin. |

The columns `risk__shortage_prob_7d` and `risk__stockout_risk_score` remain null
placeholders in Silver. They serve as schema documentation, not as production
values.

---

## Decision 2 — Bigtable as the Prediction Store (Separate Column Family)

**Chosen:** A dedicated `predictions` column family in the existing
`inventory_features` Bigtable table.

**Schema:**

| Column qualifier | Type | Description |
|---|---|---|
| `shortage_prob_7d` | float string | P(stockout ≤ 7d) ∈ [0, 1] |
| `stockout_risk_score` | float string | continuous risk ∈ [0, 1] |
| `model_version` | string | e.g. `v20260620_180918` |
| `prediction_timestamp` | RFC3339 | UTC timestamp of the prediction run |

Row key: `{warehouse_id}#{sku_id}` — unchanged from Phase 3.

**GC policy:** Keep last 3 cell versions. This preserves two prior prediction
rounds alongside the current one, enabling champion/challenger rollback inspection
without unbounded storage growth.

**Rationale for separate column family over a separate table:**

The row key, instance, and cluster are already provisioned. A separate table adds
operational overhead (split key configuration, GC policy duplication, separate IAM
scope) without benefit. Sharing the table but isolating the column family keeps
row key semantics stable and allows a single Bigtable scan to return both features
and the latest prediction.

---

## Decision 3 — Client-Side Feature Scaling

**Chosen:** `batch_scorer.py` applies `StandardScaler.transform()` (loaded from
`preprocessor.pkl`) before calling the Vertex AI Endpoint. The endpoint receives
already-scaled numeric features.

**Rejected:** Loading `preprocessor.pkl` inside the Vertex AI serving container.

**Rationale:** The scaler is a dataset artifact fit on the training split, not
part of the model graph. Loading it into the serving container would couple a data
preprocessing artifact to the model registry lifecycle. The endpoint's ONNX model
expects pre-scaled inputs — this contract is explicit in `feature_metadata.json`.

---

## Decision 4 — Serving Format: Custom FastAPI Container Wrapping ONNX Runtime

**Chosen:** Custom serving container that loads `model.onnx` with `onnxruntime`
and exposes a Vertex AI-compatible `/predict` REST endpoint.

**Evaluated alternatives:**

| Option | Verdict |
|---|---|
| Vertex AI pre-built containers (TF, PyTorch, sklearn) | None support ONNX natively |
| Triton Inference Server | Supported on Vertex AI but requires `config.pbtxt`, KFServing V2 protocol, and a 7 GB container image for a 21 KB model. Complexity is not justified. |
| Vertex AI Custom Prediction Routine (CPR) | Reduces boilerplate but is Vertex AI-proprietary; not reproducible locally. |
| **Custom FastAPI + ONNX Runtime** | ~80 lines, portable, explicit `/health` + `/predict` contract, local smoke test via `curl`. |

**Container inputs:**
- `numeric_features`: float32, shape `[batch, 21]` (pre-scaled)
- `warehouse_ids`: int64, shape `[batch]`
- `sku_ids`: int64, shape `[batch]`

**Container outputs:**
- `logit_prob`: float32 logit, shape `[batch, 1]` — sigmoidd client-side to `shortage_prob_7d`
- `risk_score`: float32 sigmoid output, shape `[batch, 1]` — used directly as `stockout_risk_score`

---

## Decision 5 — Prediction Results via Pub/Sub (Decoupled Write)

**Chosen:** After calling the endpoint, `batch_scorer.py` publishes each
prediction as a JSON message to the `prediction-results` Pub/Sub topic. A
separate `prediction_writer.py` subscribes and writes to Bigtable.

**Message schema:**
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

**Rationale:** Pub/Sub message acknowledgement ensures no prediction is lost if
the Bigtable write fails. Future consumers (alerting, dashboards) subscribe to
the same topic without touching the scorer. The dead-letter topic captures
messages that fail 5 delivery attempts for forensic inspection.

---

## Decision 6 — Champion/Challenger Gating

**Chosen:** A new model version is only promoted to 100% traffic after
`promote_model.py` verifies it outperforms the current champion on the held-out
test split across all three metrics: AUPRC, AUC-ROC, Brier.

**Mechanism:** Vertex AI Endpoint traffic splitting by model version. New versions
deploy at 0% traffic; promotion sets the split to 100% new / 0% old. Old versions
remain registered (not deleted) for rollback.

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

## Decision 8 — PSI-Based Retraining Trigger (Implementation)

**Chosen:** `drift_monitor.py` computes PSI using the training distribution
statistics stored in `preprocessor.pkl` as the reference. Current feature values
are read from Bigtable rather than Silver, matching the exact feature values that
were presented to the model at inference.

**Why Bigtable rather than Silver for drift reference:**

The model is served using features read from Bigtable. If Silver and Bigtable
diverge (e.g., a Phase 3 Spark job runs but Bigtable write fails for some rows),
the model's actual input distribution is the Bigtable distribution, not the Silver
distribution. Monitoring Silver would give a false-negative in this scenario.

**PSI computation:**

```
PSI = Σ (p_actual - p_expected) × ln(p_actual / p_expected)
```

over 10 equal-width bins derived from the training distribution. Bins with zero
observed count use a floor of 0.0001 to avoid log(0).

**Monitoring output:** `drift_monitor.py` can optionally emit custom Cloud
Monitoring metrics (`custom.googleapis.com/scip/drift/feature_psi` with label
`feature_name`, plus `prob_std`, `score_std`, `baseline_agreement_rate`, and
`retraining_required`) via `--emit-cloud-monitoring`. `infrastructure/gcp/
vertex_ai.tf` grants `scip-vertex-sa` the `monitoring.metricWriter` IAM role
needed to write them.

**Not yet implemented:** no `google_monitoring_alert_policy` Terraform
resource exists. The metrics above can be written and graphed, but nothing
pages or alerts on PSI ≥ 0.2 (or any other threshold) today — that would
require adding an alert policy resource querying this custom metric.

---

## Consequences

**Positive:**
- Silver is protected as a stable, replayable feature source.
- Predictions are versioned with per-cell history enabling rollback inspection.
- The Pub/Sub layer makes prediction delivery observable and failure-safe.
- The champion/challenger gate prevents a regressed model from reaching production.
- Distribution shift is a first-class, observable, alertable condition rather than
  a silent failure mode.

**Accepted trade-offs:**
- Two-step write (Pub/Sub → Bigtable) adds ~1–3s latency to batch scoring.
  Acceptable for a batch job running at most daily.
- The serving container must be built and pushed to Artifact Registry. One-time
  per major model architecture change.
- PSI monitors feature drift only, not label drift. Label drift monitoring (has
  the true shortage rate changed?) requires ground-truth labels with an observation
  delay matching the 7-day horizon. This is a Phase 6 concern.
- The model trained on simulated data will fail PSI checks on real Silver data.
  This is the intended behavior: the monitoring layer exists to detect and surface
  this condition, not to suppress it.
