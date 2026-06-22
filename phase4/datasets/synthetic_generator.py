"""
Realistic supply-chain history generator — additive capability only.

Produces (feature_rows, inv_events) with the same column contract as
BronzeTemporalSampler.build(), enabling DatasetBuilder.build_from_synthetic()
to reuse the existing label-generation, feature-engineering, split, and
scaling pipeline without modification.

The existing Bronze -> Spark -> Silver production pipeline is unchanged.
This module exists solely as a realistic training-data source.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Warehouse profiles ────────────────────────────────────────────────────────
# (wh_id, demand_multiplier, reorder_aggressiveness, reliability_modifier)
# reorder_aggressiveness in [0, 1]: higher → order sooner → less stockout risk
_WAREHOUSES: list[tuple] = [
    ("W-001", 2.0, 0.85, 1.02),  # large DC, well-managed
    ("W-002", 1.0, 0.65, 1.00),  # baseline regional
    ("W-003", 0.5, 0.40, 0.98),  # small depot, passive reordering
    ("W-004", 1.5, 0.90, 1.03),  # high-velocity, aggressive reorder
    ("W-005", 0.3, 0.50, 0.99),  # slow specialty
    ("W-006", 0.8, 0.30, 0.95),  # surge overflow, lowest discipline
]

# ── SKU profiles ──────────────────────────────────────────────────────────────
# (sku_id, base_demand/day, demand_cv, base_lead_time_days, lead_time_cv,
#  base_reliability, seasonality_pattern, category)
_SKUS: list[tuple] = [
    # FMCG — short lead times, high reliability, moderate demand CV
    ("SKU-001", 150.0, 0.25,  5, 0.15, 0.95, "holiday",     "FMCG"),
    ("SKU-002", 300.0, 0.20,  4, 0.12, 0.97, "flat",        "FMCG"),
    ("SKU-003",  80.0, 0.30,  6, 0.18, 0.93, "summer",      "FMCG"),
    ("SKU-004", 200.0, 0.22,  5, 0.14, 0.96, "flat",        "FMCG"),
    ("SKU-005", 120.0, 0.28,  7, 0.20, 0.94, "holiday",     "FMCG"),
    # Electronics — medium lead times, lower reliability, higher demand CV
    ("SKU-006",  25.0, 0.40, 18, 0.25, 0.79, "holiday",     "Electronics"),
    ("SKU-007",  15.0, 0.35, 21, 0.22, 0.77, "flat",        "Electronics"),
    ("SKU-008",  40.0, 0.32, 14, 0.20, 0.82, "holiday",     "Electronics"),
    ("SKU-009",  10.0, 0.50, 21, 0.30, 0.75, "flat",        "Electronics"),
    ("SKU-010",  30.0, 0.38, 18, 0.28, 0.80, "holiday",     "Electronics"),
    # Industrial parts — long lead times, lowest reliability
    ("SKU-011",   8.0, 0.20, 30, 0.18, 0.70, "q2",          "Industrial"),
    ("SKU-012",   5.0, 0.25, 45, 0.22, 0.68, "flat",        "Industrial"),
    ("SKU-013",  15.0, 0.18, 25, 0.15, 0.73, "q2",          "Industrial"),
    ("SKU-014",   3.0, 0.30, 45, 0.25, 0.67, "flat",        "Industrial"),
    ("SKU-015",  20.0, 0.22, 30, 0.20, 0.72, "q2",          "Industrial"),
    # Seasonal — high demand CV, medium lead times
    ("SKU-016",  60.0, 0.60, 10, 0.28, 0.82, "summer",      "Seasonal"),
    ("SKU-017",  90.0, 0.55, 12, 0.32, 0.84, "winter",      "Seasonal"),
    ("SKU-018",  45.0, 0.65, 10, 0.30, 0.80, "holiday",     "Seasonal"),
    ("SKU-019",  75.0, 0.50, 12, 0.28, 0.85, "summer",      "Seasonal"),
    ("SKU-020",  55.0, 0.70, 14, 0.35, 0.81, "winter",      "Seasonal"),
]

# Exported for use in generate_data.py and show_predictions.py
SKU_CATEGORIES: dict[str, str] = {row[0]: row[7] for row in _SKUS}


@dataclass
class SyntheticConfig:
    """All tunable parameters for the supply-chain simulation."""
    start_date:    str = "2025-04-01"   # ISO date of first simulation day
    total_days:    int = 402            # 30 warmup + 365 sample + 7 lookahead
    min_lookback:  int = 30             # velocity window; matches BronzeTemporalSampler
    min_lookahead: int = 7              # label horizon; matches model horizon_days
    seed:          int = 42


class SyntheticDataGenerator:
    """
    Generates (feature_rows, inv_events) matching BronzeTemporalSampler.build().

    The returned DataFrames are drop-in replacements for the GCS Bronze path
    inside DatasetBuilder.build_from_synthetic(), reusing generate_labels(),
    engineer_features(), the 70/15/15 split, and StandardScaler unchanged.

    Usage
    -----
        gen      = SyntheticDataGenerator(SyntheticConfig(seed=42))
        feat, ev = gen.build()
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self._cfg   = config or SyntheticConfig()
        self._start = date.fromisoformat(self._cfg.start_date)

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all entity simulations and return the two DataFrames.

        Returns
        -------
        feature_rows : pd.DataFrame
            One row per (warehouse_id, sku_id, sample_date).
            Columns: warehouse_id, sku_id, sample_date, current_qty,
            reorder_point, safety_stock, sales_7d, sales_30d,
            avg_daily_sales_7d, avg_daily_sales_30d, velocity_trend,
            stddev_daily_7d, lead_time_days, lead_time_stddev,
            supplier_reliability, active_po_qty, in_transit_qty,
            demand_signal_count_24h.
        inv_events : pd.DataFrame
            Daily inventory event log with columns:
            warehouse_id, sku_id, quantity_after (float), ts (UTC datetime).
            Passed to label_generator.generate_labels() unchanged.
        """
        cfg = self._cfg
        all_feat: list[dict] = []
        all_evts: list[dict] = []

        n_entities = len(_WAREHOUSES) * len(_SKUS)
        logger.info(
            "Simulating %d days × %d entities (%d potential rows)",
            cfg.total_days, n_entities, cfg.total_days * n_entities,
        )

        for wh_idx, (wh_id, wh_mult, wh_reorder, wh_rel_mod) in enumerate(_WAREHOUSES):
            for sku_idx, sku_params in enumerate(_SKUS):
                entity_seed = cfg.seed * 10_000 + wh_idx * 100 + sku_idx
                sim = _EntitySimulator(
                    wh_id      = wh_id,
                    sku_params = sku_params,
                    wh_mult    = wh_mult,
                    wh_reorder = wh_reorder,
                    wh_rel_mod = wh_rel_mod,
                    start_date = self._start,
                    total_days = cfg.total_days,
                    lookback   = cfg.min_lookback,
                    lookahead  = cfg.min_lookahead,
                    rng        = np.random.default_rng(entity_seed),
                )
                feat, evts = sim.run()
                all_feat.extend(feat)
                all_evts.extend(evts)

        feature_rows = pd.DataFrame(all_feat)
        inv_events   = pd.DataFrame(all_evts)
        inv_events["ts"] = pd.to_datetime(inv_events["ts"], utc=True)

        n_dates = feature_rows["sample_date"].nunique() if len(feature_rows) else 0
        logger.info(
            "Simulation complete: %d feature rows (%d unique dates), "
            "%d inventory events",
            len(feature_rows), n_dates, len(inv_events),
        )
        return feature_rows, inv_events


# ── Entity simulator ──────────────────────────────────────────────────────────

class _EntitySimulator:
    """Day-by-day inventory simulation for one (warehouse_id, sku_id) pair."""

    def __init__(
        self,
        wh_id:      str,
        sku_params: tuple,
        wh_mult:    float,
        wh_reorder: float,
        wh_rel_mod: float,
        start_date: date,
        total_days: int,
        lookback:   int,
        lookahead:  int,
        rng:        np.random.Generator,
    ) -> None:
        (sku_id, base_demand, demand_cv, base_lt, lt_cv,
         base_rel, season_pat, _cat) = sku_params

        self._wh_id    = wh_id
        self._sku_id   = sku_id
        self._start    = start_date
        self._total    = total_days
        self._lookback = lookback
        self._lookahead = lookahead
        self._rng      = rng
        self._season   = season_pat

        # Effective demand parameters for this entity
        self._daily_mean  = base_demand * wh_mult
        self._demand_cv   = demand_cv
        # Gamma: shape = 1/CV², scale = mean/shape (so E[X]=mean, Var[X]=mean²·CV²)
        self._gshape = max(1.0 / max(demand_cv ** 2, 1e-6), 0.5)
        self._gscale = self._daily_mean / self._gshape

        # Supplier parameters; reliability clamped to a sensible range
        self._base_lt    = float(base_lt)
        self._lt_cv      = lt_cv
        self._reliability = float(np.clip(base_rel * wh_rel_mod, 0.50, 0.999))

        # Inventory policy
        # coverage_mult ∈ [0.95, 1.85]: lower for undisciplined warehouses
        coverage_mult       = 0.5 + wh_reorder * 1.5
        self._reorder_point = self._daily_mean * self._base_lt * coverage_mult
        self._safety_stock  = self._daily_mean * self._base_lt * 0.5
        self._target_qty    = self._daily_mean * self._base_lt * 3.0

        # Initial state: start well-stocked
        self._qty = self._target_qty

        # [arrival_day, qty, placed_day, promised_lt, on_time_flag]
        self._pending: list[list] = []

        # (completion_day, promised_lt, actual_lt, on_time_flag) for 90-day rolling stats
        self._completed: list[tuple[int, float, float, bool]] = []

        # (placed_day, qty) for active_po_qty rolling sum
        self._po_placed: list[tuple[int, float]] = []

        # Fulfilled daily demand; used for velocity rolling windows
        self._demand_hist = np.zeros(total_days, dtype=np.float64)

        # Pre-build promotional spike schedule
        self._spikes = self._build_spikes()

    # ── Spike schedule ────────────────────────────────────────────────────────

    def _build_spikes(self) -> dict[int, float]:
        """Poisson spike process at λ=0.03/day (~11 events/year per entity)."""
        spikes: dict[int, float] = {}
        day = 0
        while day < self._total:
            gap  = int(self._rng.exponential(scale=1.0 / 0.03)) + 1
            day += gap
            if day >= self._total:
                break
            duration = int(self._rng.integers(1, 4))   # 1–3 days
            mult     = float(self._rng.uniform(1.8, 4.5))
            for d in range(day, min(day + duration, self._total)):
                spikes[d] = max(spikes.get(d, 1.0), mult)
            day += duration
        return spikes

    # ── Seasonality ───────────────────────────────────────────────────────────

    @staticmethod
    def _seasonal_mult(doy: int, pattern: str) -> float:
        """Demand multiplier for day-of-year [1..365] and seasonality pattern."""
        t = (doy - 1) / 364.0   # normalised to [0.0, 1.0]
        if pattern == "holiday":
            # Sharp peak centred on Dec 1 (t ≈ 0.92)
            return 1.0 + 0.80 * math.exp(-0.5 * ((t - 0.92) / 0.07) ** 2)
        if pattern == "summer":
            # Peak at Jul 20 (t ≈ 0.55)
            return 1.0 + 0.40 * math.exp(-0.5 * ((t - 0.55) / 0.12) ** 2)
        if pattern == "winter":
            # Twin peaks at Jan 1 (t=0) and Dec 20 (t=0.97)
            p1 = math.exp(-0.5 * (t / 0.06) ** 2)
            p2 = math.exp(-0.5 * ((t - 1.0) / 0.06) ** 2)
            return 1.0 + 0.60 * max(p1, p2)
        if pattern == "q2":
            # Plateau peak at May 15 (t ≈ 0.37)
            return 1.0 + 0.25 * math.exp(-0.5 * ((t - 0.37) / 0.12) ** 2)
        return 1.0   # flat

    # ── Core simulation loop ──────────────────────────────────────────────────

    def run(self) -> tuple[list[dict], list[dict]]:
        feature_rows: list[dict] = []
        inv_events:   list[dict] = []

        for day_idx in range(self._total):
            sim_date = self._start + timedelta(days=day_idx)

            # 1. Receive any POs due on or before today
            still_pending = []
            for po in self._pending:
                arrival_day, qty, placed_day, promised_lt, on_time_flag = po
                if day_idx >= arrival_day:
                    self._qty += qty
                    actual_lt  = float(day_idx - placed_day)
                    self._completed.append((day_idx, promised_lt, actual_lt, on_time_flag))
                    inv_events.append({
                        "warehouse_id":  self._wh_id,
                        "sku_id":        self._sku_id,
                        "quantity_after": float(self._qty),
                        "ts": pd.Timestamp(sim_date, tz="UTC").replace(
                            hour=8, minute=0, second=0, microsecond=0
                        ),
                    })
                else:
                    still_pending.append(po)
            self._pending = still_pending

            # 2. Sample demand and apply it
            doy        = sim_date.timetuple().tm_yday
            seas_mult  = self._seasonal_mult(doy, self._season)
            promo_mult = self._spikes.get(day_idx, 1.0)
            adj_mean   = self._daily_mean * seas_mult * promo_mult
            # Re-parameterise Gamma for adjusted mean (CV unchanged)
            adj_scale  = adj_mean / self._gshape
            demand     = float(self._rng.gamma(self._gshape, adj_scale))

            units_sold         = min(demand, self._qty)
            self._qty         -= units_sold
            self._demand_hist[day_idx] = units_sold

            # End-of-day event (quantity_after == 0 signals a stockout to label_generator)
            inv_events.append({
                "warehouse_id":  self._wh_id,
                "sku_id":        self._sku_id,
                "quantity_after": float(self._qty),
                "ts": pd.Timestamp(sim_date, tz="UTC").replace(
                    hour=23, minute=59, second=0, microsecond=0
                ),
            })

            # 3. Place one PO if below reorder point and none outstanding
            if self._qty < self._reorder_point and not self._pending:
                self._place_po(day_idx)

            # 4. Record feature snapshot only for usable sample days
            if self._lookback <= day_idx < self._total - self._lookahead:
                feature_rows.append(self._snapshot(day_idx, sim_date))

        return feature_rows, inv_events

    def _place_po(self, day_idx: int) -> None:
        """Place a single replenishment PO."""
        order_qty   = max(
            self._target_qty - self._qty,
            self._daily_mean * self._base_lt,
        )
        lt_std      = self._base_lt * self._lt_cv
        promised_lt = float(np.clip(
            self._rng.normal(self._base_lt, lt_std),
            1.0, self._base_lt * 4.0,
        ))
        on_time    = bool(self._rng.random() < self._reliability)
        delay      = 0.0 if on_time else float(self._rng.poisson(lam=3.0))
        actual_lt  = promised_lt + delay
        arrival    = day_idx + max(1, int(math.ceil(actual_lt)))

        self._pending.append([arrival, order_qty, day_idx, promised_lt, on_time])
        self._po_placed.append((day_idx, order_qty))

    def _snapshot(self, day_idx: int, sim_date: date) -> dict:
        """Compute all 15 raw Silver features for this (entity, day)."""
        # Velocity: rolling windows over fulfilled daily demand
        s7  = float(self._demand_hist[max(0, day_idx - 6)  : day_idx + 1].sum())
        s30 = float(self._demand_hist[max(0, day_idx - 29) : day_idx + 1].sum())
        n7  = min(day_idx + 1, 7)
        n30 = min(day_idx + 1, 30)
        avg7  = s7  / n7
        avg30 = s30 / n30
        trend = (avg7 / avg30) if avg30 > 1e-6 else 1.0
        win7  = self._demand_hist[max(0, day_idx - 6) : day_idx + 1]
        std7  = float(np.std(win7, ddof=0)) if n7 > 1 else 0.0

        # Supplier stats: 90-day rolling window of completed POs
        recent = [
            (p_lt, a_lt, ot)
            for (comp_day, p_lt, a_lt, ot) in self._completed
            if day_idx - comp_day <= 90
        ]
        if recent:
            a_arr      = np.array([a for _, a, _ in recent], dtype=np.float64)
            lt_mean    = float(np.mean(a_arr))
            lt_std_val = float(np.std(a_arr, ddof=0)) if len(a_arr) > 1 else self._base_lt * self._lt_cv * 0.5
            on_time_rt = float(np.mean([ot for _, _, ot in recent]))
        else:
            lt_mean    = self._base_lt
            lt_std_val = self._base_lt * self._lt_cv * 0.5
            on_time_rt = self._reliability

        # PO pipeline: active_po_qty = POs placed in last 90 days (mirrors Bronze)
        active_po  = sum(qty for (pd_day, qty) in self._po_placed if day_idx - pd_day <= 90)
        # in_transit = pending POs placed in last 30 days (mirrors SHIPPED events)
        in_transit = sum(po[1] for po in self._pending if day_idx - po[2] <= 30)

        # Demand signal count: proxy for true demand activity independent of stockouts
        doy        = sim_date.timetuple().tm_yday
        sig_mult   = self._seasonal_mult(doy, self._season) * self._spikes.get(day_idx, 1.0)
        sig_count  = float(max(0.0, round(self._daily_mean * sig_mult / 10.0)))

        return {
            "warehouse_id":            self._wh_id,
            "sku_id":                  self._sku_id,
            "sample_date":             sim_date,
            # Inventory state
            "current_qty":             float(self._qty),
            "reorder_point":           float(self._reorder_point),
            "safety_stock":            float(self._safety_stock),
            # Velocity (7d and 30d rolling)
            "sales_7d":                s7,
            "sales_30d":               s30,
            "avg_daily_sales_7d":      avg7,
            "avg_daily_sales_30d":     avg30,
            "velocity_trend":          trend,
            "stddev_daily_7d":         std7,
            # Supplier (90-day rolling from completed POs)
            "lead_time_days":          lt_mean,
            "lead_time_stddev":        lt_std_val,
            "supplier_reliability":    on_time_rt,
            # PO pipeline
            "active_po_qty":           float(active_po),
            "in_transit_qty":          float(in_transit),
            # Demand signals
            "demand_signal_count_24h": sig_count,
        }
