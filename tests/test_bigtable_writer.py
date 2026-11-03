"""
spark/utils/bigtable_writer.py: what one partition sends to Bigtable. The table
is a fake, so this needs neither Spark nor the emulator;
tests/test_silver_bigtable.py covers the same path against the emulator.
"""
import struct
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "spark"))

from config.feature_config import FEATURE_REGISTRY  # noqa: E402
from utils import bigtable_writer as bw  # noqa: E402

_COLUMNS = [f"{family}__{attr}" for family, attrs in FEATURE_REGISTRY.items() for attr in attrs]


class _FakeRow:
    def __init__(self, key):
        self.row_key = key
        self.set_cells, self.deleted = {}, []

    def set_cell(self, column_family_id, column, value, timestamp):
        self.set_cells[(column_family_id, column)] = value

    def delete_cell(self, column_family_id, column):
        self.deleted.append((column_family_id, column))


class _FakeTable:
    def __init__(self, status_code=0, rpc_error=None):
        self.rows, self._code, self._rpc_error = [], status_code, rpc_error

    def direct_row(self, key):
        return _FakeRow(key)

    def mutate_rows(self, batch):
        if self._rpc_error:
            raise self._rpc_error
        self.rows += batch
        return [types.SimpleNamespace(code=self._code, message="fake") for _ in batch]


def _write(monkeypatch, table, row):
    client = types.SimpleNamespace(
        instance=lambda _: types.SimpleNamespace(table=lambda _: table),
        close=lambda: None,
    )
    from google.cloud import bigtable
    monkeypatch.setattr(bigtable, "Client", lambda **_: client)
    bw._write_partition(iter([row]), "project", "instance", "table")


def _row(**values):
    row = dict.fromkeys(_COLUMNS, 1.0)
    row["row_key"] = "W-001#SKU-001"
    row.update(values)
    return row


def test_a_null_deletes_the_cell_instead_of_leaving_the_previous_value(monkeypatch):
    table = _FakeTable()
    _write(monkeypatch, table, _row(supplier__in_transit_qty=None))

    written = table.rows[0]
    assert ("supplier", b"transit_qty") in written.deleted
    assert ("supplier", b"transit_qty") not in written.set_cells
    assert written.set_cells[("supplier", b"po_qty")] == struct.pack(">d", 1.0)
    assert len(written.deleted) == 1


def test_a_rejected_row_fails_the_partition(monkeypatch):
    with pytest.raises(RuntimeError, match="failed for 1 of 1 rows"):
        _write(monkeypatch, _FakeTable(status_code=14), _row())


def test_a_failed_rpc_fails_the_partition(monkeypatch):
    with pytest.raises(RuntimeError, match="failed for 1 of 1 rows"):
        _write(monkeypatch, _FakeTable(rpc_error=OSError("unavailable")), _row())
