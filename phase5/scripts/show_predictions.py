"""
show_predictions.py - Supply Chain Intelligence Platform risk dashboard.

Reads the latest predictions from the Bigtable predictions column family
and renders a terminal dashboard showing prediction variance, per-warehouse
risk tables, per-SKU-category breakdowns, highest/lowest risk entities,
and a risk heatmap.

Requires no UI libraries beyond the Python standard library.

Usage
-----
    python phase5/scripts/show_predictions.py \\
        --project  supply-chain-platform-499920 \\
        --instance scip-feature-store \\
        --sa-key   infrastructure/gcp/keys/scip-prediction-writer-sa.json

    # Minimal (uses defaults):
    python phase5/scripts/show_predictions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# -- SKU category mapping (mirrors synthetic_generator._SKUS) -----------------
# Hard-coded so show_predictions.py has zero dependency on phase4 modules.
_SKU_CATEGORIES: dict[str, str] = {
    "SKU-001": "FMCG",        "SKU-002": "FMCG",
    "SKU-003": "FMCG",        "SKU-004": "FMCG",
    "SKU-005": "FMCG",
    "SKU-006": "Electronics", "SKU-007": "Electronics",
    "SKU-008": "Electronics", "SKU-009": "Electronics",
    "SKU-010": "Electronics",
    "SKU-011": "Industrial",  "SKU-012": "Industrial",
    "SKU-013": "Industrial",  "SKU-014": "Industrial",
    "SKU-015": "Industrial",
    "SKU-016": "Seasonal",    "SKU-017": "Seasonal",
    "SKU-018": "Seasonal",    "SKU-019": "Seasonal",
    "SKU-020": "Seasonal",
}

_CATEGORY_ORDER = ["FMCG", "Electronics", "Industrial", "Seasonal"]

_HIGH_RISK_THRESHOLD = 0.50
_VARIANCE_COLLAPSE_STD = 0.01


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCIP risk dashboard - reads predictions from Bigtable"
    )
    p.add_argument(
        "--project",
        default="supply-chain-platform-499920",
        help="GCP project ID (default: supply-chain-platform-499920)",
    )
    p.add_argument(
        "--instance",
        default="scip-feature-store",
        help="Bigtable instance ID (default: scip-feature-store)",
    )
    p.add_argument(
        "--table",
        default="inventory_features",
        help="Bigtable table ID (default: inventory_features)",
    )
    p.add_argument(
        "--sa-key",
        default="infrastructure/gcp/keys/scip-prediction-writer-sa.json",
        help="Service account key path",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of highest/lowest risk entities to display (default: 5)",
    )
    return p.parse_args()


# -- Bigtable read -------------------------------------------------------------

def read_predictions(project: str, instance_id: str, table_id: str, sa_key: str) -> list[dict]:
    """Read all prediction rows from Bigtable. Returns list of dicts."""
    from google.cloud import bigtable
    from google.oauth2 import service_account

    try:
        creds  = service_account.Credentials.from_service_account_file(sa_key)
        client = bigtable.Client(project=project, credentials=creds, admin=False)
    except Exception as exc:
        print(f"\n  [ERROR] Failed to authenticate: {exc}")
        print(f"  Key path: {sa_key}")
        sys.exit(1)

    table = client.instance(instance_id).table(table_id)

    records: list[dict] = []
    try:
        rows = table.read_rows()
        for row in rows:
            key  = row.row_key.decode("utf-8")
            preds = row.cells.get("predictions", {})

            prob_cells  = preds.get(b"shortage_prob_7d", [])
            score_cells = preds.get(b"stockout_risk_score", [])
            ver_cells   = preds.get(b"model_version", [])
            ts_cells    = preds.get(b"prediction_timestamp", [])

            if not prob_cells or not score_cells:
                continue

            try:
                prob  = float(prob_cells[0].value.decode("utf-8"))
                score = float(score_cells[0].value.decode("utf-8"))
            except (ValueError, AttributeError):
                continue

            version = ver_cells[0].value.decode("utf-8")  if ver_cells  else "unknown"
            ts      = ts_cells[0].value.decode("utf-8")   if ts_cells   else "unknown"

            # Row key format: {warehouse_id}#{sku_id}
            parts = key.split("#", 1)
            if len(parts) != 2:
                continue
            wh_id, sku_id = parts

            records.append({
                "warehouse_id":        wh_id,
                "sku_id":              sku_id,
                "category":            _SKU_CATEGORIES.get(sku_id, "Unknown"),
                "shortage_prob_7d":    prob,
                "stockout_risk_score": score,
                "model_version":       version,
                "prediction_timestamp": ts,
            })
    except Exception as exc:
        print(f"\n  [ERROR] Failed to read from Bigtable: {exc}")
        sys.exit(1)
    finally:
        client.close()

    return records


# -- Formatting helpers --------------------------------------------------------

def _bar(value: float, scale: float = 0.40, width: int = 4) -> str:
    """ASCII bar proportional to value. scale=0.40 -> value=1.0 gives 40 blocks."""
    n = max(0, min(width, int(round(value * scale * width / 1.0))))
    return "#" * n + " " * (width - n)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


# -- Dashboard sections --------------------------------------------------------

def _section(title: str) -> None:
    print()
    print(f"  {title}")
    print("  " + "-" * 60)


def render_dashboard(records: list[dict], top_n: int) -> None:
    if not records:
        print("\n  No predictions found. Run score.py first.")
        return

    probs  = [r["shortage_prob_7d"]    for r in records]
    scores = [r["stockout_risk_score"] for r in records]
    n      = len(records)

    mean_prob  = sum(probs)  / n
    std_prob   = _stdev(probs,  mean_prob)
    mean_score = sum(scores) / n
    std_score  = _stdev(scores, mean_score)
    collapsed  = std_prob < _VARIANCE_COLLAPSE_STD

    version = records[0]["model_version"]
    ts      = records[0]["prediction_timestamp"]

    # -- Header -------------------------------------------------------------
    print()
    print("  +" + "=" * 62 + "+")
    print("  |     SUPPLY CHAIN INTELLIGENCE PLATFORM - RISK DASHBOARD     |")
    print(f"  |     Model {version:<20}  {n} entities scored     |")
    print("  +" + "=" * 62 + "+")

    # -- Overall summary -----------------------------------------------------
    _section("OVERALL SUMMARY")
    variance_status = (
        "COLLAPSED - retrain model (see drift_monitor.py)"
        if collapsed else
        "OK - meaningful variation detected  [PASS]"
    )
    print(f"    Entities scored            : {n}")
    print(f"    Mean shortage_prob_7d      : {_fmt(mean_prob)}")
    print(f"    Std  shortage_prob_7d      : {_fmt(std_prob)}")
    print(f"    Mean stockout_risk_score   : {_fmt(mean_score)}")
    print(f"    Std  stockout_risk_score   : {_fmt(std_score)}")
    print(f"    Variance status            : {variance_status}")
    print(f"    Last scored                : {ts}")

    # -- By warehouse ---------------------------------------------------------
    _section("RISK BY WAREHOUSE")
    print(f"    {'Warehouse':<10}  {'Entities':>8}  {'Mean Risk':>9}  {'Max Risk':>8}  {'High-Risk (>0.50)':>17}")
    print("    " + "-" * 58)
    wh_groups: dict[str, list[dict]] = {}
    for r in records:
        wh_groups.setdefault(r["warehouse_id"], []).append(r)
    for wh in sorted(wh_groups):
        grp     = wh_groups[wh]
        g_probs = [r["shortage_prob_7d"] for r in grp]
        g_mean  = sum(g_probs) / len(g_probs)
        g_max   = max(g_probs)
        g_high  = sum(1 for p in g_probs if p > _HIGH_RISK_THRESHOLD)
        bar     = _bar(g_mean)
        print(f"    {wh:<10}  {len(grp):>8}  {_fmt(g_mean):>9}  {_fmt(g_max):>8}  {g_high:>17}  {bar}")

    # -- By SKU category -------------------------------------------------------
    _section("RISK BY SKU CATEGORY")
    print(f"    {'Category':<14}  {'SKUs':>4}  {'Mean Risk':>9}  {'Max Risk':>8}")
    print("    " + "-" * 42)
    cat_groups: dict[str, list[dict]] = {}
    for r in records:
        cat_groups.setdefault(r["category"], []).append(r)
    for cat in _CATEGORY_ORDER:
        grp = cat_groups.get(cat, [])
        if not grp:
            continue
        g_probs = [r["shortage_prob_7d"] for r in grp]
        g_mean  = sum(g_probs) / len(g_probs)
        g_max   = max(g_probs)
        bar     = _bar(g_mean)
        print(f"    {cat:<14}  {len(grp):>4}  {_fmt(g_mean):>9}  {_fmt(g_max):>8}  {bar}")

    # -- Top N highest risk ----------------------------------------------------
    _section(f"TOP {top_n} HIGHEST RISK ENTITIES")
    print(f"    {'#':>2}  {'Entity':<22}  {'shortage_prob_7d':>16}  {'stockout_risk_score':>19}")
    print("    " + "-" * 64)
    top = sorted(records, key=lambda r: r["shortage_prob_7d"], reverse=True)[:top_n]
    for i, r in enumerate(top, 1):
        entity = f"{r['warehouse_id']} / {r['sku_id']}"
        bar    = _bar(r["shortage_prob_7d"], scale=0.30, width=8)
        print(f"    {i:>2}  {entity:<22}  {_fmt(r['shortage_prob_7d']):>16}  {_fmt(r['stockout_risk_score']):>19}  {bar}")

    # -- Top N lowest risk -----------------------------------------------------
    _section(f"TOP {top_n} LOWEST RISK ENTITIES")
    print(f"    {'#':>2}  {'Entity':<22}  {'shortage_prob_7d':>16}  {'stockout_risk_score':>19}")
    print("    " + "-" * 64)
    bottom = sorted(records, key=lambda r: r["shortage_prob_7d"])[:top_n]
    for i, r in enumerate(bottom, 1):
        entity = f"{r['warehouse_id']} / {r['sku_id']}"
        print(f"    {i:>2}  {entity:<22}  {_fmt(r['shortage_prob_7d']):>16}  {_fmt(r['stockout_risk_score']):>19}")

    # -- Risk heatmap ----------------------------------------------------------
    _section("RISK HEATMAP  (shortage_prob_7d avg by warehouse x SKU category)")
    cat_abbr = {"FMCG": "FMCG", "Electronics": "Elec.", "Industrial": "Indus.", "Seasonal": "Season."}
    header_cats = [cat_abbr.get(c, c) for c in _CATEGORY_ORDER if c in cat_groups]
    header = "    " + " " * 12 + "  ".join(f"{c:>7}" for c in header_cats)
    print(header)
    print("    " + "-" * (12 + 9 * len(header_cats)))

    for wh in sorted(wh_groups):
        grp = wh_groups[wh]
        cells: list[str] = []
        wh_bar_val = 0.0
        for cat in _CATEGORY_ORDER:
            if cat not in cat_groups:
                continue
            cat_probs = [r["shortage_prob_7d"] for r in grp if r["category"] == cat]
            if cat_probs:
                cell_mean = sum(cat_probs) / len(cat_probs)
                wh_bar_val = max(wh_bar_val, cell_mean)
                cells.append(f"{cell_mean:>7.2f}")
            else:
                cells.append("   --  ")
        bar = _bar(wh_bar_val, scale=0.25, width=4)
        print(f"    {wh:<10}  {'  '.join(cells)}  {bar}")

    print()
    print("    bar scale: 1 block ~ 0.10 risk")
    print()
    print("  " + "=" * 62)
    print()


# -- Statistics helpers --------------------------------------------------------

def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    records = read_predictions(
        project     = args.project,
        instance_id = args.instance,
        table_id    = args.table,
        sa_key      = args.sa_key,
    )
    render_dashboard(records, top_n=args.top_n)


if __name__ == "__main__":
    main()
