"""
show_feature_sample.py - SCIP feature sample dashboard.

Reads a representative sample of rows from the Bigtable feature store,
decodes the raw feature values, and renders a per-entity card display.
Designed as a demo step showing the pipeline narrative:

    Feature Data  ->  Model Scoring  ->  Risk Dashboard

Column families read:
    inventory  : current_qty (cur_qty), reorder_point (reorder_pt), safety_stock (safety_stk)
    velocity   : sales_7d, sales_30d, avg_daily_sales_7d (avg_d_7d)
    demand     : demand_signal_count_24h (sig_24h)
    supplier   : lead_time_days (lt_days), supplier_reliability (reliability),
                 active_po_qty (po_qty), in_transit_qty (transit_qty)
    predictions: shortage_prob_7d, stockout_risk_score (if already scored)

Usage
-----
    python phase5/scripts/show_feature_sample.py
    python phase5/scripts/show_feature_sample.py --n 8 --sort warehouse
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

# ---------------------------------------------------------------------------
# Bigtable column mapping
#   (family, abbreviated_qualifier) -> display_name
# Derived from feature_client._BIGTABLE_TO_FEATURE and the Bigtable write
# contract in spark/utils/bigtable_writer.py.
# ---------------------------------------------------------------------------
_FEATURE_COLS: dict[tuple[str, str], str] = {
    ("inventory", "cur_qty"):     "current_qty",
    ("inventory", "reorder_pt"):  "reorder_point",
    ("inventory", "safety_stk"):  "safety_stock",
    ("velocity",  "sales_7d"):    "sales_7d",
    ("velocity",  "sales_30d"):   "sales_30d",
    ("velocity",  "avg_d_7d"):    "avg_daily_sales_7d",
    ("demand",    "sig_24h"):     "demand_signal_count_24h",
    ("supplier",  "lt_days"):     "lead_time_days",
    ("supplier",  "reliability"): "supplier_reliability",
    ("supplier",  "po_qty"):      "active_po_qty",
    ("supplier",  "transit_qty"): "in_transit_qty",
}

_FEATURE_FAMILIES = frozenset({"inventory", "velocity", "demand", "supplier"})
_ALL_FAMILIES     = _FEATURE_FAMILIES | {"predictions"}

_HIGH_RISK_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCIP feature sample dashboard — shows raw feature data + predictions"
    )
    p.add_argument("--project",  default="supply-chain-platform-499920")
    p.add_argument("--instance", default="scip-feature-store")
    p.add_argument("--table",    default="inventory_features")
    p.add_argument(
        "--sa-key",
        default="infrastructure/gcp/keys/scip-prediction-writer-sa.json",
    )
    p.add_argument(
        "--n",
        type=int,
        default=7,
        help="Number of entities to display (default: 7)",
    )
    p.add_argument(
        "--features-only",
        action="store_true",
        help="Show input feature data only — omit risk labels and model output",
    )
    p.add_argument(
        "--sort",
        choices=["risk", "warehouse", "all"],
        default="risk",
        help=(
            "risk     - top-N by shortage_prob_7d (default, requires predictions)\n"
            "warehouse - alphabetical by warehouse / SKU\n"
            "all       - show top-4 highest + top-3 lowest risk (requires predictions)"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Bigtable read
# ---------------------------------------------------------------------------

def _decode_double(value: bytes) -> float | None:
    if len(value) == 8:
        return struct.unpack(">d", value)[0]
    try:
        return float(value.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def read_all_rows(
    project: str,
    instance_id: str,
    table_id: str,
    sa_key: str,
) -> list[dict]:
    """
    Scan the entire inventory_features table.

    Returns a list of dicts, one per valid row key, containing:
        warehouse_id, sku_id, plus all decoded feature and prediction fields.
    Predictions are included only if the 'predictions' column family has been
    written by score.py; missing values are stored as None.
    """
    from google.cloud import bigtable
    from google.cloud.bigtable import row_filters
    from google.oauth2 import service_account

    try:
        creds  = service_account.Credentials.from_service_account_file(sa_key)
        client = bigtable.Client(project=project, credentials=creds, admin=False)
    except Exception as exc:
        print(f"\n  [ERROR] Authentication failed: {exc}")
        print(f"  Key path: {sa_key}")
        sys.exit(1)

    table = client.instance(instance_id).table(table_id)

    filter_ = row_filters.RowFilterChain(filters=[
        row_filters.FamilyNameRegexFilter("|".join(sorted(_ALL_FAMILIES))),
        row_filters.CellsColumnLimitFilter(1),
    ])

    records: list[dict] = []
    try:
        for row_data in table.read_rows(filter_=filter_):
            row_key = row_data.row_key.decode("utf-8")
            parts   = row_key.split("#", 1)
            if len(parts) != 2:
                continue

            warehouse_id, sku_id = parts
            rec: dict = {"warehouse_id": warehouse_id, "sku_id": sku_id}

            for cf_name, cf_data in row_data.cells.items():
                if cf_name in _FEATURE_FAMILIES:
                    for col_bytes, cells in cf_data.items():
                        col    = col_bytes.decode("utf-8")
                        fname  = _FEATURE_COLS.get((cf_name, col))
                        if fname:
                            val = _decode_double(cells[0].value)
                            if val is not None:
                                rec[fname] = val

                elif cf_name == "predictions":
                    for col_bytes, cells in cf_data.items():
                        col = col_bytes.decode("utf-8")
                        if col in ("shortage_prob_7d", "stockout_risk_score"):
                            try:
                                rec[col] = float(cells[0].value.decode("utf-8"))
                            except (ValueError, UnicodeDecodeError):
                                pass

            records.append(rec)
    finally:
        client.close()

    return records


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _has_predictions(records: list[dict]) -> bool:
    return any("shortage_prob_7d" in r for r in records)


def select_sample(records: list[dict], n: int, sort: str) -> list[dict]:
    if sort == "warehouse":
        ordered = sorted(records, key=lambda r: (r["warehouse_id"], r["sku_id"]))
        return ordered[:n]

    if sort == "all" and _has_predictions(records):
        scored   = sorted(
            (r for r in records if "shortage_prob_7d" in r),
            key=lambda r: r["shortage_prob_7d"],
            reverse=True,
        )
        top_high = n // 2 + (n % 2)
        top_low  = n // 2
        selected = scored[:top_high] + scored[-top_low:]
        # deduplicate while preserving order
        seen: set[str] = set()
        unique: list[dict] = []
        for r in selected:
            key = f"{r['warehouse_id']}#{r['sku_id']}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:n]

    # default: "risk" (or fallback if no predictions)
    if _has_predictions(records):
        scored = sorted(
            (r for r in records if "shortage_prob_7d" in r),
            key=lambda r: r["shortage_prob_7d"],
            reverse=True,
        )
        return scored[:n]

    # no predictions: spread across warehouses
    by_wh: dict[str, list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["warehouse_id"], r["sku_id"])):
        by_wh.setdefault(r["warehouse_id"], []).append(r)
    selected: list[dict] = []
    wh_lists = list(by_wh.values())
    i = 0
    while len(selected) < n and any(wh_lists):
        bucket = wh_lists[i % len(wh_lists)]
        if bucket:
            selected.append(bucket.pop(0))
        i += 1
    return selected


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fv(rec: dict, key: str, fmt: str = ".0f", fallback: str = "  --") -> str:
    val = rec.get(key)
    if val is None:
        return fallback
    return format(val, fmt)


def _risk_tag(rec: dict) -> str:
    prob = rec.get("shortage_prob_7d")
    if prob is None:
        return ""
    if prob >= _HIGH_RISK_THRESHOLD:
        return "  [HIGH RISK]"
    if prob >= 0.15:
        return "  [ELEVATED]"
    return "  [LOW RISK]"


def render(records: list[dict], n_total: int, features_only: bool = False) -> None:
    has_preds = _has_predictions(records)

    print()
    print("  " + "=" * 62)
    print("  SUPPLY CHAIN INTELLIGENCE PLATFORM")
    if features_only:
        print(f"  Feature Sample -- Input Data Only  ({len(records)} of {n_total} entities)")
    else:
        print(f"  Feature Sample  ({len(records)} of {n_total} entities)")
        if has_preds:
            sort_note = "sorted by shortage_prob_7d desc"
        else:
            sort_note = "no predictions yet -- run score.py first"
        print(f"  [{sort_note}]")
    print("  " + "=" * 62)

    for i, rec in enumerate(records):
        wh  = rec["warehouse_id"]
        sku = rec["sku_id"]
        tag = "" if features_only else _risk_tag(rec)

        print()
        print(f"  {wh} / {sku}{tag}")
        print("  " + "-" * 50)

        # -- Inventory
        cur_qty   = _fv(rec, "current_qty")
        reorder   = _fv(rec, "reorder_point")
        safety    = _fv(rec, "safety_stock")
        lt        = _fv(rec, "lead_time_days")
        print(f"  Inventory")
        print(f"    Current Qty        : {cur_qty:>8}    Reorder Point : {reorder:>8}")
        print(f"    Safety Stock       : {safety:>8}    Lead Time     : {lt:>5} days")

        # -- Demand / velocity
        s7d   = _fv(rec, "sales_7d")
        s30d  = _fv(rec, "sales_30d")
        ads   = _fv(rec, "avg_daily_sales_7d", fmt=".1f")
        sigs  = _fv(rec, "demand_signal_count_24h")
        print(f"  Demand & Velocity")
        print(f"    Sales 7d           : {s7d:>8}    Sales 30d     : {s30d:>8}")
        print(f"    Avg Daily Sales    : {ads:>8}    Demand Sigs   : {sigs:>8}")

        # -- Supply-side
        rel     = _fv(rec, "supplier_reliability", fmt=".3f")
        po_qty  = _fv(rec, "active_po_qty")
        transit = _fv(rec, "in_transit_qty")
        print(f"  Supply")
        print(f"    Reliability        : {rel:>8}    Active PO Qty : {po_qty:>8}")
        print(f"    In-Transit Qty     : {transit:>8}")

        # -- Predictions (if scored and not suppressed)
        if not features_only:
            prob  = rec.get("shortage_prob_7d")
            score = rec.get("stockout_risk_score")
            if prob is not None:
                print(f"  Model Output")
                print(f"    Shortage Prob 7d   : {prob:>8.3f}    Risk Score    : {score:>8.3f}")
            else:
                print(f"  Model Output       : (not yet scored)")

        if i < len(records) - 1:
            print()
            print("  " + "- " * 26)

    print()
    print("  " + "=" * 62)
    if features_only:
        print("  -> run score.py to generate updated predictions")
        print("  -> run show_predictions.py to view risk results")
    else:
        print("  -> run score.py to update model predictions")
        print("  -> run show_predictions.py for the full risk dashboard")
    print("  " + "=" * 62)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    records = read_all_rows(
        project     = args.project,
        instance_id = args.instance,
        table_id    = args.table,
        sa_key      = args.sa_key,
    )

    if not records:
        print("\n  No rows found. Verify the Bigtable table name and credentials.")
        sys.exit(1)

    sample = select_sample(records, args.n, args.sort)
    render(sample, n_total=len(records), features_only=args.features_only)


if __name__ == "__main__":
    main()
