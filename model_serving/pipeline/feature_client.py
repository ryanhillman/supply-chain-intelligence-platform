"""
feature_client.py — reads all feature rows from the Bigtable inventory_features table.

Design
------
- Scans all rows in the table (no row key filter) to score every entity.
- Restricts to feature column families (inventory, velocity, demand, supplier)
  plus lineage:silver_ts, the ISO-8601 time of the Silver run that wrote the
  row. The risk and predictions families are skipped — they contain ML
  outputs, not model inputs. Callers use split_fresh() to drop rows whose
  Silver write is older than their freshness threshold: feature families keep
  only the latest cell (no age-based GC), so a quiet store would otherwise be
  scored on stale features.
- Numeric cells are stored as big-endian IEEE 754 doubles (8 bytes).
  See spark/utils/bigtable_writer.py for the write side.
- Abbreviated column qualifiers are mapped to flat feature names via
  _BIGTABLE_TO_FEATURE, derived from spark/config/feature_config.py FEATURE_REGISTRY.
- Only the latest cell version per column is read (CellsColumnLimitFilter(1)).

Row key format: {warehouse_id}#{sku_id}  e.g. "W-001#SKU-005"

Required packages:
    google-cloud-bigtable>=2.22.0
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Maps (bigtable_family, abbreviated_qualifier) -> flat feature name.
# Derived from spark/config/feature_config.py FEATURE_REGISTRY and
# spark/utils/bigtable_writer.py encoding contract.
# String columns (last_evt, last_upd, last_sig) are intentionally absent;
# they cannot be decoded as doubles and are not model inputs.
_BIGTABLE_TO_FEATURE: dict[tuple[str, str], str] = {
    ("inventory", "cur_qty"):      "current_qty",
    ("inventory", "reorder_pt"):   "reorder_point",
    ("inventory", "safety_stk"):   "safety_stock",
    ("velocity",  "sales_7d"):     "sales_7d",
    ("velocity",  "sales_30d"):    "sales_30d",
    ("velocity",  "avg_d_7d"):     "avg_daily_sales_7d",
    ("velocity",  "avg_d_30d"):    "avg_daily_sales_30d",
    ("velocity",  "v_trend"):      "velocity_trend",
    ("velocity",  "stddev_7d"):    "stddev_daily_7d",
    ("demand",    "sig_24h"):      "demand_signal_count_24h",
    ("supplier",  "lt_days"):      "lead_time_days",
    ("supplier",  "lt_std"):       "lead_time_stddev",
    ("supplier",  "reliability"):  "supplier_reliability",
    ("supplier",  "po_qty"):       "active_po_qty",
    ("supplier",  "transit_qty"):  "in_transit_qty",
}

# Column families that hold model input features (not outputs or metadata).
_FEATURE_FAMILIES: frozenset[str] = frozenset({"inventory", "velocity", "demand", "supplier"})

# Lineage cell stamping the Silver run that wrote the row (UTF-8 ISO-8601; see
# spark/config/feature_config.py FEATURE_REGISTRY["lineage"]).
_LINEAGE_FAMILY:      str = "lineage"
_SILVER_TS_QUALIFIER: str = "silver_ts"


@dataclass
class FeatureRow:
    """One entity's decoded Bigtable row.

    Attributes
    ----------
    row_key:
        Raw Bigtable row key string ("W-001#SKU-005").
    warehouse_id:
        Warehouse component of the row key ("W-001").
    sku_id:
        SKU component of the row key ("SKU-005").
    features:
        Flat feature name -> float value, covering the 15 raw model inputs
        that are present in Bigtable. Missing features will be filled with
        domain defaults by RequestBuilder before inference.
    last_silver_write:
        UTC time of the Silver run that last wrote this row (lineage:silver_ts),
        or None if the cell is missing or unparseable.
    """

    row_key:           str
    warehouse_id:      str
    sku_id:            str
    features:          dict[str, float] = field(default_factory=dict)
    last_silver_write: datetime | None  = None


class FeatureClient:
    """
    Reads feature rows from the Bigtable inventory_features table.

    Parameters
    ----------
    project_id:
        GCP project ID.
    instance_id:
        Bigtable instance ID (e.g. "scip-feature-store").
    table_id:
        Bigtable table ID (default "inventory_features").
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        instance_id: str,
        table_id: str = "inventory_features",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id  = project_id
        self.instance_id = instance_id
        self.table_id    = table_id

        from google.cloud import bigtable

        client_kwargs: dict = {"project": project_id, "admin": False}
        if credentials_path:
            from google.oauth2 import service_account
            client_kwargs["credentials"] = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        self._client = bigtable.Client(**client_kwargs)
        self._table  = self._client.instance(instance_id).table(table_id)

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan_all(self) -> list[FeatureRow]:
        """
        Scan every row in the inventory_features table and return decoded feature rows.

        Reads the feature column families (inventory, velocity, demand, supplier)
        and lineage:silver_ts.
        Only reads the latest cell version per column qualifier.

        Returns
        -------
        list[FeatureRow]
            One entry per valid row key. Rows with unexpected key format are skipped.
            Feature dicts may contain fewer than 15 keys if some columns are absent
            for a given entity; RequestBuilder fills defaults for missing keys.

        Raises
        ------
        google.api_core.exceptions.GoogleAPIError
            On network failure or Bigtable unavailability. Propagated to caller.
        """
        from google.cloud.bigtable import row_filters

        filter_ = row_filters.RowFilterChain(filters=[
            row_filters.FamilyNameRegexFilter("|".join(sorted(_FEATURE_FAMILIES | {_LINEAGE_FAMILY}))),
            row_filters.CellsColumnLimitFilter(1),
        ])

        logger.info(
            "Scanning Bigtable %s/%s (project=%s, families=%s)",
            self.instance_id, self.table_id, self.project_id,
            sorted(_FEATURE_FAMILIES | {_LINEAGE_FAMILY}),
        )

        rows: list[FeatureRow] = []
        n_cells_total = 0
        n_skipped     = 0

        for row_data in self._table.read_rows(filter_=filter_):
            row_key = row_data.row_key.decode("utf-8")
            parts   = row_key.split("#")

            if len(parts) != 2:
                logger.warning("Unexpected row key format %r — skipping", row_key)
                n_skipped += 1
                continue

            warehouse_id, sku_id = parts[0], parts[1]
            raw_features: dict[str, float] = {}
            last_silver_write: datetime | None = None

            for cf_name, cf_data in row_data.cells.items():
                if cf_name == _LINEAGE_FAMILY:
                    ts_cells = cf_data.get(_SILVER_TS_QUALIFIER.encode("utf-8"))
                    if ts_cells:
                        last_silver_write = _decode_timestamp(ts_cells[0].value, row_key)
                    continue
                if cf_name not in _FEATURE_FAMILIES:
                    continue

                for col_qualifier_bytes, cells in cf_data.items():
                    col_qualifier = col_qualifier_bytes.decode("utf-8")
                    feature_name  = _BIGTABLE_TO_FEATURE.get((cf_name, col_qualifier))

                    if feature_name is None:
                        # String qualifier (last_evt, last_upd, etc.) or unknown column.
                        continue

                    cell_value = cells[0].value
                    decoded    = _decode_double(cell_value, cf_name, col_qualifier)

                    if decoded is not None:
                        raw_features[feature_name] = decoded
                        n_cells_total += 1

            rows.append(FeatureRow(
                row_key=row_key,
                warehouse_id=warehouse_id,
                sku_id=sku_id,
                features=raw_features,
                last_silver_write=last_silver_write,
            ))

            if len(rows) % 20 == 0:
                logger.info("  scanned %d rows so far ...", len(rows))

        logger.info(
            "Scan complete: %d rows, %d feature cells decoded, %d rows skipped",
            len(rows), n_cells_total, n_skipped,
        )

        return rows

    def close(self) -> None:
        """Close the underlying Bigtable client connection."""
        self._client.close()

    def __enter__(self) -> "FeatureClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Module helpers ─────────────────────────────────────────────────────────────

def split_fresh(
    rows: list[FeatureRow],
    max_age_hours: float | None,
    now: datetime | None = None,
) -> tuple[list[FeatureRow], int]:
    """
    Split rows into (fresh_rows, n_stale) by the age of lineage:silver_ts.

    A row is stale when last_silver_write is missing or older than
    max_age_hours. None or a non-positive max_age_hours disables the check
    and returns every row as fresh.
    """
    if max_age_hours is None or max_age_hours <= 0:
        return rows, 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(hours=max_age_hours)
    fresh  = [
        r for r in rows
        if r.last_silver_write is not None and r.last_silver_write >= cutoff
    ]
    return fresh, len(rows) - len(fresh)


def _decode_timestamp(value: bytes, row_key: str) -> datetime | None:
    """Decode a UTF-8 ISO-8601 cell to an aware UTC datetime, or None."""
    try:
        ts = datetime.fromisoformat(value.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        logger.warning(
            "Row %s: cannot parse lineage:silver_ts %r — treating as stale",
            row_key, value[:40],
        )
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _decode_double(value: bytes, family: str, qualifier: str) -> float | None:
    """
    Decode a big-endian IEEE 754 double (8 bytes) to float.

    Bigtable cells are written with struct.pack(">d", float(value)) by
    spark/utils/bigtable_writer.py. An 8-byte value is always a double.
    A non-8-byte value is logged as a warning and returns None so the
    caller can skip it and let RequestBuilder fill the domain default.
    """
    if len(value) == 8:
        return struct.unpack(">d", value)[0]

    # Graceful fallback for cells stored as UTF-8 strings (non-standard writes).
    try:
        return float(value.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        logger.warning(
            "Cannot decode cell (%s, %s) with %d bytes as double or UTF-8 — skipping",
            family, qualifier, len(value),
        )
        return None
