"""
feature_registry: the one definition of the Bigtable layout and cell encoding
that the Spark writer, FeatureClient, RequestBuilder, the feature dashboard and
the training code share.
"""
import struct

from feature_registry import (
    FEATURE_REGISTRY,
    RAW_FEATURE_CELLS,
    RAW_FEATURE_COLS,
    RAW_FEATURE_FAMILIES,
    STRING_FEATURES,
    decode_double,
    encode_cell,
)

_COLUMNS = {f"{family}__{feature}" for family, features in FEATURE_REGISTRY.items() for feature in features}


def test_every_model_input_has_exactly_one_numeric_cell():
    assert sorted(RAW_FEATURE_CELLS.values()) == sorted(RAW_FEATURE_COLS)
    assert len(set(RAW_FEATURE_COLS)) == len(RAW_FEATURE_COLS) == 15
    assert RAW_FEATURE_FAMILIES == {"inventory", "velocity", "demand", "supplier"}
    for (family, _), feature in RAW_FEATURE_CELLS.items():
        assert f"{family}__{feature}" not in STRING_FEATURES


def test_string_features_are_registry_columns():
    assert STRING_FEATURES <= _COLUMNS
    assert len(_COLUMNS) == 29


def test_numbers_round_trip_as_big_endian_doubles():
    cell = encode_cell("velocity__sales_7d", 12.5)
    assert cell == struct.pack(">d", 12.5)
    assert decode_double(cell) == 12.5
    assert decode_double(encode_cell("risk__reorder_urgency", 3)) == 3.0


def test_strings_and_nulls():
    assert encode_cell("inventory__last_event_type", "SALE") == b"SALE"
    assert encode_cell("velocity__sales_7d", None) is None
    assert encode_cell("velocity__sales_7d", "not a number") is None


def test_decode_accepts_a_utf8_number_and_rejects_anything_else():
    assert decode_double(b"0.25") == 0.25
    assert decode_double(b"SALE") is None
    assert decode_double(b"\xff\xfe") is None
