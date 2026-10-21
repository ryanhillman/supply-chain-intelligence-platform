"""
BronzeTemporalSampler — reads all five Bronze GCS topics and builds a
feature DataFrame of (warehouse_id, sku_id, sample_date) rows.

One row per (warehouse, sku, calendar day) within the usable date range:
  start = first_inventory_event_date + min_lookback_days
  end   = last_inventory_event_date  - min_lookahead_days

The lookahead buffer ensures label_generator.py has 7 days of future events
for every row. The lookback buffer ensures velocity windows (30d) are fully
populated at the earliest sample date.

Feature groups computed here (raw values only — derived features added by
feature_engineer.py):
  inventory  : current_qty, reorder_point, safety_stock
  velocity   : sales_7d, sales_30d, avg_daily_sales_7d, avg_daily_sales_30d,
               velocity_trend, stddev_daily_7d
  supplier   : lead_time_days, lead_time_stddev, supplier_reliability
  po         : active_po_qty  (CREATED/SUBMITTED in last 90d)
  shipment   : in_transit_qty (SHIPPED in last 30d)
  demand     : demand_signal_count_24h
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import gcsfs
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from pyarrow.fs import FSSpecHandler, PyFileSystem

logger = logging.getLogger(__name__)

# Bronze partition columns injected by Parquet directory structure.
# We keep them during read (PyArrow decodes them automatically) but they are
# not used for feature computation — the `timestamp` string field is used instead.
_PARTITION_COLS = {"year", "month", "day", "hour"}

# Field subsets to load per topic (avoids reading unused columns from GCS).
_INVENTORY_COLS = [
    "warehouse_id", "sku_id", "event_type",
    "quantity_after", "quantity_delta",
    "reorder_point", "safety_stock", "timestamp",
]
_SUPPLIER_COLS = [
    "warehouse_id", "lead_time_days", "on_time_delivery_rate", "timestamp",
]
_PO_COLS = [
    "warehouse_id", "sku_id", "event_type", "quantity_ordered", "timestamp",
]
_SHIPMENT_COLS = [
    "warehouse_id", "sku_id", "event_type", "quantity", "timestamp",
]
_DEMAND_COLS = [
    "warehouse_id", "sku_id", "timestamp",
]


class BronzeTemporalSampler:
    """
    Reads Bronze GCS Parquet and builds a feature row per (warehouse, sku, day).

    Parameters
    ----------
    bucket:
        GCS bucket name, e.g. "my-project-scip-lake".
    credentials_path:
        Path to a service account JSON key, or None to use Application
        Default Credentials.
    min_lookback_days:
        Days of history required before the first sample date (default 30,
        matches the 30d velocity window in silver_feature_engineering.py).
    min_lookahead_days:
        Days of future data required after the last sample date (default 7,
        matches the shortage prediction horizon).
    """

    def __init__(
        self,
        bucket: str,
        credentials_path: str | None = None,
        min_lookback_days: int = 30,
        min_lookahead_days: int = 7,
        simulate_days: int = 0,
    ) -> None:
        self._bucket = bucket
        self._credentials_path = credentials_path
        self._min_lookback = min_lookback_days
        self._min_lookahead = min_lookahead_days
        self._simulate_days = simulate_days
        self._fs = gcsfs.GCSFileSystem(
            token=credentials_path if credentials_path else "google_default"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all Bronze topics and build the feature DataFrame.

        Returns
        -------
        features : pd.DataFrame
            One row per (warehouse_id, sku_id, sample_date).
            Columns: all raw feature groups listed in the module docstring.
        inv_events : pd.DataFrame
            Full inventory event log with parsed timestamps.
            Passed to label_generator.py to avoid re-reading from GCS.
        """
        logger.info("Loading Bronze inventory events from gs://%s/bronze/", self._bucket)
        inv = self._read_topic("inventory.events", _INVENTORY_COLS)
        inv = self._parse_ts(inv)

        span_days = (inv["ts"].max() - inv["ts"].min()).days
        required = self._min_lookback + self._min_lookahead + 1
        if span_days < required:
            if self._simulate_days > 0:
                logger.warning(
                    "Bronze data spans only %d day(s); need %d. "
                    "Tiling events across %d simulated days for training.",
                    span_days, required, self._simulate_days,
                )
                inv = self._tile_events(inv, self._simulate_days)
            else:
                raise ValueError(
                    f"Bronze data spans only {span_days} day(s); need {required}. "
                    "Pass --simulate-days N (e.g. 60) to tile the events for training."
                )

        sample_dates = self._sample_date_range(inv)
        if len(sample_dates) == 0:
            raise ValueError(
                "Bronze data does not span enough days for training. "
                f"Need at least {required} days of events."
            )
        logger.info(
            "Sample date range: %s → %s (%d dates, %d pairs)",
            sample_dates[0].date(), sample_dates[-1].date(), len(sample_dates),
            inv.groupby(["warehouse_id", "sku_id"]).ngroups,
        )

        logger.info("Computing inventory snapshot features")
        snapshots = self._snapshot_features(inv, sample_dates)

        logger.info("Computing velocity features")
        velocity = self._velocity_features(inv, sample_dates)

        logger.info("Loading supplier events")
        sup = self._read_topic("suppliers.events", _SUPPLIER_COLS)
        sup = self._parse_ts(sup)
        supplier = self._supplier_features(sup, sample_dates)

        logger.info("Loading purchase order events")
        po_raw = self._read_topic("purchase.orders.events", _PO_COLS)
        po_raw = self._parse_ts(po_raw)
        po = self._po_features(po_raw, sample_dates)

        logger.info("Loading shipment events")
        shp_raw = self._read_topic("shipments.events", _SHIPMENT_COLS)
        shp_raw = self._parse_ts(shp_raw)
        shipment = self._shipment_features(shp_raw, sample_dates)

        logger.info("Loading demand signals")
        dem_raw = self._read_topic("demand.signals", _DEMAND_COLS)
        dem_raw = self._parse_ts(dem_raw)
        demand = self._demand_features(dem_raw, sample_dates)

        features = (
            snapshots
            .merge(velocity,  on=["warehouse_id", "sku_id", "sample_date"], how="left")
            .merge(supplier,  on=["warehouse_id", "sample_date"],            how="left")
            .merge(po,        on=["warehouse_id", "sku_id", "sample_date"],  how="left")
            .merge(shipment,  on=["warehouse_id", "sku_id", "sample_date"],  how="left")
            .merge(demand,    on=["warehouse_id", "sku_id", "sample_date"],  how="left")
        )

        features["active_po_qty"]           = features["active_po_qty"].fillna(0.0)
        features["in_transit_qty"]          = features["in_transit_qty"].fillna(0.0)
        features["demand_signal_count_24h"] = features["demand_signal_count_24h"].fillna(0.0)
        features["lead_time_days"]          = features["lead_time_days"].fillna(14.0)
        features["lead_time_stddev"]        = features["lead_time_stddev"].fillna(0.0)
        features["supplier_reliability"]    = features["supplier_reliability"].fillna(0.9)

        logger.info("Feature DataFrame built: %d rows, %d columns", len(features), len(features.columns))
        return features, inv

    # ── GCS I/O ───────────────────────────────────────────────────────────────

    def _read_topic(self, topic: str, columns: list[str]) -> pd.DataFrame:
        try:
            # gcsfs.ls() returns the directory itself as the first entry, which
            # confuses pyarrow's file-discovery walk (it tries to open_input_file
            # on a directory path and raises FileNotFoundError). Glob for explicit
            # file paths instead so pyarrow never has to walk the directory.
            all_paths = self._fs.glob(f"{self._bucket}/bronze/{topic}/**/*.parquet")
            file_paths = [p for p in all_paths if "_temporary" not in p]
            if not file_paths:
                raise FileNotFoundError(
                    f"No committed Parquet files found for Bronze topic {topic}"
                )
            logger.info("Discovered %d files for topic %s", len(file_paths), topic)
            pa_fs = PyFileSystem(FSSpecHandler(self._fs))
            dataset = pads.dataset(
                file_paths,
                filesystem=pa_fs,
                format="parquet",
                partitioning="hive",
            )
            available = {f.name for f in dataset.schema}
            requested = [c for c in columns if c in available]
            table = dataset.to_table(columns=requested)
            df = table.to_pandas()
            logger.info("Read %s: %d rows", topic, len(df))
            return df
        except Exception as exc:
            logger.error("Failed to read Bronze topic %s: %s", topic, exc)
            raise

    @staticmethod
    def _parse_ts(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
        """Parse ISO-8601 string timestamps to UTC-aware datetime, stored as 'ts'."""
        df = df.copy()
        df["ts"] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df.dropna(subset=["ts"])

    # ── Date range ────────────────────────────────────────────────────────────

    def _sample_date_range(self, inv: pd.DataFrame) -> pd.DatetimeIndex:
        first = inv["ts"].min().date()
        last  = inv["ts"].max().date()
        start = first + timedelta(days=self._min_lookback)
        end   = last  - timedelta(days=self._min_lookahead)
        if start > end:
            return pd.DatetimeIndex([])
        return pd.date_range(start=start, end=end, freq="D", tz="UTC")

    # ── Feature groups ────────────────────────────────────────────────────────

    def _snapshot_features(
        self, inv: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Latest inventory state per (warehouse, sku) as of end-of-day on each
        sample date. Uses np.searchsorted for O(log N) per-date lookup.
        """
        inv_sorted = inv.sort_values("ts")
        rows: list[dict] = []

        for (wh, sku), grp in inv_sorted.groupby(["warehouse_id", "sku_id"], sort=False):
            ts_ns  = grp["ts"].values.astype("int64")        # nanoseconds
            qa_arr = grp["quantity_after"].values.astype(float)
            rp_arr = grp["reorder_point"].fillna(0).values.astype(float)
            ss_arr = grp["safety_stock"].fillna(0).values.astype(float)

            for d in sample_dates:
                eod_ns = int((d + pd.Timedelta(days=1)).value)
                pos = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                if pos == 0:
                    continue
                idx = pos - 1
                rows.append({
                    "warehouse_id": wh,
                    "sku_id":       sku,
                    "sample_date":  d.date(),
                    "current_qty":  qa_arr[idx],
                    "reorder_point": rp_arr[idx],
                    "safety_stock": ss_arr[idx],
                })

        return pd.DataFrame(rows)

    def _velocity_features(
        self, inv: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Rolling 7d and 30d sales velocity per (warehouse, sku, date).
        Demand is measured as -quantity_delta for SALE events (stock decreases).

        avg_daily_sales_{7,30}d divide by the ACTUAL number of days observed
        before each sample_date, not hardcoded 7.0/30.0. The class-level
        min_lookback_days=30 buffer only guarantees 30 days have passed since
        the GLOBAL earliest event across every (warehouse, sku) pair -- it
        does not guarantee any given pair's own first sale is that old (a
        pair with sparse or late-starting history can still be sampled
        during its individual cold start). Dividing by a fixed constant in
        that case manufactures a velocity_trend artifact exactly like the
        one fixed in spark/jobs/silver_feature_engineering.py's
        _compute_velocity: when sales_7d == sales_30d (no sales before the
        last 7 days), trend becomes a numeric artifact of the window-length
        ratio rather than a real trend. This matters more here than in the
        live Spark job, since this is the TRAINING label generator -- the
        artifact would be baked into the model's training data itself.
        """
        sales = inv[inv["event_type"] == "SALE"].copy()
        sales["units_sold"] = (-sales["quantity_delta"]).clip(lower=0).astype(float)
        sales_sorted = sales.sort_values("ts")
        rows: list[dict] = []
        _NS_PER_DAY = 24 * 60 * 60 * 1_000_000_000

        for (wh, sku), grp in sales_sorted.groupby(["warehouse_id", "sku_id"], sort=False):
            ts_ns   = grp["ts"].values.astype("int64")
            units   = grp["units_sold"].values
            first_ts_ns = int(ts_ns[0]) if len(ts_ns) > 0 else None
            # Pre-compute daily totals for stddev (avoids repeated groupby inside loop)
            grp["date_only"] = grp["ts"].dt.floor("D")
            daily   = grp.groupby("date_only")["units_sold"].sum()
            d_index = daily.index.values.astype("int64")  # ns
            d_vals  = daily.values

            for d in sample_dates:
                eod_ns   = int((d + pd.Timedelta(days=1)).value)
                ns_7d    = int((d - pd.Timedelta(days=6)).value)
                ns_30d   = int((d - pd.Timedelta(days=29)).value)

                end_pos  = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                pos_7d   = int(np.searchsorted(ts_ns, ns_7d,  side="left"))
                pos_30d  = int(np.searchsorted(ts_ns, ns_30d, side="left"))

                s7  = float(units[pos_7d:end_pos].sum())
                s30 = float(units[pos_30d:end_pos].sum())

                # Daily stddev over 7d window
                d_end_ns   = int(d.value)
                d_start_ns = ns_7d
                d_mask = (d_index >= d_start_ns) & (d_index <= d_end_ns)
                daily_window = d_vals[d_mask]
                stddev = float(np.std(daily_window, ddof=0)) if len(daily_window) > 1 else 0.0

                if first_ts_ns is not None:
                    days_since_first = (eod_ns - first_ts_ns) // _NS_PER_DAY + 1
                    days_observed_30 = max(1, min(30, days_since_first))
                else:
                    days_observed_30 = 30
                days_observed_7 = min(7, days_observed_30)

                avg_d7  = s7  / days_observed_7
                avg_d30 = s30 / days_observed_30
                trend   = (avg_d7 / avg_d30) if avg_d30 > 0.0 else 1.0

                rows.append({
                    "warehouse_id":        wh,
                    "sku_id":              sku,
                    "sample_date":         d.date(),
                    "sales_7d":            s7,
                    "sales_30d":           s30,
                    "avg_daily_sales_7d":  avg_d7,
                    "avg_daily_sales_30d": avg_d30,
                    "velocity_trend":      trend,
                    "stddev_daily_7d":     stddev,
                })

        return pd.DataFrame(rows)

    def _supplier_features(
        self, sup: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        90-day rolling supplier lead time and reliability per (warehouse, date).
        Aggregated at warehouse level, matching silver_feature_engineering.py.
        """
        sup_sorted = sup.sort_values("ts")
        rows: list[dict] = []

        for wh, grp in sup_sorted.groupby("warehouse_id", sort=False):
            ts_ns  = grp["ts"].values.astype("int64")
            lt_arr = grp["lead_time_days"].values.astype(float)
            otr_arr = grp["on_time_delivery_rate"].values.astype(float)

            for d in sample_dates:
                eod_ns   = int((d + pd.Timedelta(days=1)).value)
                ns_90d   = int((d - pd.Timedelta(days=90)).value)
                end_pos  = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                pos_90d  = int(np.searchsorted(ts_ns, ns_90d, side="left"))

                lt_win  = lt_arr[pos_90d:end_pos]
                otr_win = otr_arr[pos_90d:end_pos]
                lt_valid  = lt_win[~np.isnan(lt_win)]
                otr_valid = otr_win[~np.isnan(otr_win)]

                rows.append({
                    "warehouse_id":       wh,
                    "sample_date":        d.date(),
                    "lead_time_days":     float(np.mean(lt_valid))           if len(lt_valid) > 0  else np.nan,
                    "lead_time_stddev":   float(np.std(lt_valid, ddof=0))    if len(lt_valid) > 1  else 0.0,
                    "supplier_reliability": float(np.mean(otr_valid))        if len(otr_valid) > 0 else np.nan,
                })

        return pd.DataFrame(rows)

    def _po_features(
        self, po: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Active PO quantity per (warehouse, sku, date): sum of quantity_ordered
        from CREATED or SUBMITTED events in the last 90 days.
        """
        active = po[po["event_type"].isin(["CREATED", "SUBMITTED"])].sort_values("ts")
        rows: list[dict] = []

        for (wh, sku), grp in active.groupby(["warehouse_id", "sku_id"], sort=False):
            ts_ns  = grp["ts"].values.astype("int64")
            qty    = grp["quantity_ordered"].fillna(0).values.astype(float)

            for d in sample_dates:
                eod_ns  = int((d + pd.Timedelta(days=1)).value)
                ns_90d  = int((d - pd.Timedelta(days=90)).value)
                end_pos = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                pos_90d = int(np.searchsorted(ts_ns, ns_90d, side="left"))

                rows.append({
                    "warehouse_id":  wh,
                    "sku_id":        sku,
                    "sample_date":   d.date(),
                    "active_po_qty": float(qty[pos_90d:end_pos].sum()),
                })

        return pd.DataFrame(rows)

    def _shipment_features(
        self, shp: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        In-transit quantity per (warehouse, sku, date): sum of quantity from
        SHIPPED events in the last 30 days.
        """
        shipped = shp[shp["event_type"] == "SHIPPED"].sort_values("ts")
        rows: list[dict] = []

        for (wh, sku), grp in shipped.groupby(["warehouse_id", "sku_id"], sort=False):
            ts_ns = grp["ts"].values.astype("int64")
            qty   = grp["quantity"].fillna(0).values.astype(float)

            for d in sample_dates:
                eod_ns  = int((d + pd.Timedelta(days=1)).value)
                ns_30d  = int((d - pd.Timedelta(days=30)).value)
                end_pos = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                pos_30d = int(np.searchsorted(ts_ns, ns_30d, side="left"))

                rows.append({
                    "warehouse_id":  wh,
                    "sku_id":        sku,
                    "sample_date":   d.date(),
                    "in_transit_qty": float(qty[pos_30d:end_pos].sum()),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def _tile_events(inv: pd.DataFrame, n_days: int, seed: int = 42) -> pd.DataFrame:
        """
        Tile a single day of events across n_days simulated days.

        Each tile is shifted by i*24h from the original start.  quantity_after
        and quantity_delta receive ±10% multiplicative noise per tile to produce
        realistic variation in inventory levels across the simulated window.

        This is a development-mode workaround for when Bronze data spans < 38 days.
        Production Bronze data accumulates naturally over time and will not need this.
        """
        rng = np.random.default_rng(seed)
        tiles: list[pd.DataFrame] = []

        for i in range(n_days):
            tile = inv.copy()
            tile["ts"] = tile["ts"] + pd.Timedelta(days=i)
            tile["timestamp"] = tile["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"

            noise = rng.uniform(0.90, 1.10, size=len(tile))
            for col in ("quantity_after", "quantity_delta"):
                if col in tile.columns:
                    tile[col] = (tile[col].astype(float) * noise).round().astype(
                        tile[col].dtype
                    )
            tiles.append(tile)

        tiled = pd.concat(tiles, ignore_index=True).sort_values("ts")
        logger.info(
            "Tiled events: %d rows across %d days (%s → %s)",
            len(tiled), n_days,
            tiled['ts'].min().date(), tiled['ts'].max().date(),
        )
        return tiled

    def _demand_features(
        self, dem: pd.DataFrame, sample_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Demand signal count per (warehouse, sku, date) in the 24h window
        [T-1d, T], matching the demand__demand_signal_count_24h Silver column.
        """
        dem_sorted = dem.sort_values("ts")
        rows: list[dict] = []

        for (wh, sku), grp in dem_sorted.groupby(["warehouse_id", "sku_id"], sort=False):
            ts_ns = grp["ts"].values.astype("int64")

            for d in sample_dates:
                eod_ns  = int((d + pd.Timedelta(days=1)).value)
                ns_24h  = int((d - pd.Timedelta(days=1)).value)
                end_pos = int(np.searchsorted(ts_ns, eod_ns, side="left"))
                pos_24h = int(np.searchsorted(ts_ns, ns_24h, side="left"))

                rows.append({
                    "warehouse_id":            wh,
                    "sku_id":                  sku,
                    "sample_date":             d.date(),
                    "demand_signal_count_24h": float(end_pos - pos_24h),
                })

        return pd.DataFrame(rows)
