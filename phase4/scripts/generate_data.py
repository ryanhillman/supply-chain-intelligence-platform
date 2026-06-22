"""
generate_data.py — standalone CLI for the synthetic supply-chain data generator.

Runs the simulation and writes feature_rows.parquet and inv_events.parquet to
an output directory. Use --inspect to also apply label_generator and print
the label distribution by warehouse and SKU category.

This script is optional — build_dataset.py --synthetic runs the generator
inline. Use generate_data.py when you want to inspect or tune the simulation
before committing to a full dataset build.

Usage
-----
    # Generate and write parquet files:
    python phase4/scripts/generate_data.py \\
        --total-days 402 \\
        --start-date 2025-04-01 \\
        --seed 42 \\
        --output-dir phase4/artifacts/synthetic

    # Also apply label_generator and print distribution:
    python phase4/scripts/generate_data.py ... --inspect
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from phase4.datasets.synthetic_generator import (
    SKU_CATEGORIES,
    SyntheticConfig,
    SyntheticDataGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("generate_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate realistic synthetic supply-chain training data"
    )
    p.add_argument(
        "--total-days",
        type=int,
        default=402,
        help="Total simulation days (default: 402 = 30 warmup + 365 sample + 7 lookahead)",
    )
    p.add_argument(
        "--start-date",
        default="2025-04-01",
        help="Simulation start date in ISO format (default: 2025-04-01)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--output-dir",
        default="phase4/artifacts/synthetic",
        help="Output directory for parquet files (default: phase4/artifacts/synthetic)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Apply label_generator and print positive-rate breakdown by "
            "warehouse and SKU category. Adds ~30s to runtime."
        ),
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SyntheticConfig(
        start_date = args.start_date,
        total_days = args.total_days,
        seed       = args.seed,
    )

    logger.info("Starting synthetic data generation")
    logger.info("  start_date : %s", config.start_date)
    logger.info("  total_days : %d (%.0f sample days per entity)",
                config.total_days,
                config.total_days - config.min_lookback - config.min_lookahead)
    logger.info("  seed       : %d", config.seed)
    logger.info("  output_dir : %s", out_dir)

    gen = SyntheticDataGenerator(config=config)
    feature_rows, inv_events = gen.build()

    feat_path = out_dir / "feature_rows.parquet"
    evt_path  = out_dir / "inv_events.parquet"
    feature_rows.to_parquet(feat_path, index=False)
    inv_events.to_parquet(evt_path,    index=False)

    _print_summary(feature_rows, inv_events)

    if args.inspect:
        _inspect_labels(feature_rows, inv_events)

    print(f"  feature_rows  -> {feat_path}")
    print(f"  inv_events    -> {evt_path}")
    print()
    print("  Next step — build the dataset:")
    print("    python phase4/scripts/build_dataset.py \\")
    print(f"        --synthetic \\")
    print(f"        --total-days {args.total_days} \\")
    print(f"        --start-date {args.start_date} \\")
    print(f"        --seed {args.seed} \\")
    print("        --output phase4/artifacts/dataset")
    print("=" * 62)


def _print_summary(feature_rows, inv_events) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print("  Synthetic Data Generation Complete")
    print(sep)
    print(f"  Feature rows      : {len(feature_rows):>10,}")
    print(f"  Inventory events  : {len(inv_events):>10,}")

    if len(feature_rows):
        n_dates = feature_rows["sample_date"].nunique()
        d_min   = feature_rows["sample_date"].min()
        d_max   = feature_rows["sample_date"].max()
        print(f"  Unique dates      : {n_dates:>10,}")
        print(f"  Warehouses        : {feature_rows['warehouse_id'].nunique():>10,}")
        print(f"  SKUs              : {feature_rows['sku_id'].nunique():>10,}")
        print(f"  Date range        : {d_min} -> {d_max}")
        print()

        summary_cols = [
            ("current_qty",          "Current qty"),
            ("avg_daily_sales_7d",   "Avg daily sales (7d)"),
            ("lead_time_days",       "Lead time (days)"),
            ("supplier_reliability", "Supplier reliability"),
            ("active_po_qty",        "Active PO qty"),
        ]
        print("  Feature summary (mean ± std):")
        for col, label in summary_cols:
            if col in feature_rows.columns:
                m = feature_rows[col].mean()
                s = feature_rows[col].std()
                print(f"    {label:<24} : {m:>10.2f} ± {s:>8.2f}")
        print()

    if len(inv_events):
        n_zero = int((inv_events["quantity_after"] == 0.0).sum())
        pct    = n_zero / len(inv_events) * 100
        print(f"  Zero-inventory events : {n_zero:,} / {len(inv_events):,} ({pct:.1f}%)")
    print()


def _inspect_labels(feature_rows, inv_events) -> None:
    """Apply label_generator and print label distribution breakdown."""
    from phase4.datasets.label_generator import generate_labels

    print("  Label inspection (applying label_generator, ~30s)...")
    labeled = generate_labels(feature_rows, inv_events, horizon_days=7)

    overall = labeled["stockout_label_7d"].mean()
    n_pos   = int(labeled["stockout_label_7d"].sum())
    n_total = len(labeled)
    print(f"  Overall positive rate : {overall:.3f}  ({n_pos:,} / {n_total:,} rows)")
    print()

    print("  Positive rate by warehouse:")
    for wh, grp in labeled.groupby("warehouse_id"):
        rate = grp["stockout_label_7d"].mean()
        bar  = "#" * max(1, int(rate * 40))
        print(f"    {wh}   {rate:.3f}  {bar}")
    print()

    labeled["category"] = labeled["sku_id"].map(SKU_CATEGORIES)
    print("  Positive rate by SKU category:")
    for cat, grp in labeled.groupby("category"):
        rate = grp["stockout_label_7d"].mean()
        bar  = "#" * max(1, int(rate * 40))
        print(f"    {cat:<14}  {rate:.3f}  {bar}")
    print()


if __name__ == "__main__":
    main()
