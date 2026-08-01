"""Offline unit tests for src.preprocess.clean.

Uses a temporary directory as a fake PROJECT_ROOT with a hand-built
manifest + raw batch file, so no network access or real BOLD data is
required.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from src.preprocess.clean import clean

GOOD_SEQ = "ACGT" * 150  # 600bp, well within the default [500, 750] band


def _write_batch(raw_dir, name, records):
    path = raw_dir / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh)
    return path


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    raw_dir = tmp_path / "data" / "raw" / "bold_test"
    raw_dir.mkdir(parents=True)
    return tmp_path, raw_dir


def _base_config():
    return {
        "output": {
            "raw_dir": "data/raw",
            "manifest_path": "data/manifest.json",
        },
        "cleaning": {
            "min_length": 500,
            "max_length": 750,
            "max_ambiguous_fraction": 0.01,
            "output_path": "data/processed/sequences.parquet",
        },
    }


def _write_manifest(tmp_path, batch_files):
    manifest = {"batch_files": [str(p) for p in batch_files]}
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)


def test_clean_drops_missing_required_fields(project_dir):
    tmp_path, raw_dir = project_dir
    batch_path = _write_batch(
        raw_dir,
        "batch_0000.json",
        [
            {
                "species": "Gadus morhua",
                "genus": "Gadus",
                "family": "Gadidae",
                "order": "Gadiformes",
                "nuc": GOOD_SEQ,
                "processid": "P1",
                "bin_uri": "BOLD:AAA0001",
            },
            {
                "species": None,
                "genus": "Gadus",
                "family": "Gadidae",
                "order": "Gadiformes",
                "nuc": GOOD_SEQ,
            },
        ],
    )
    _write_manifest(tmp_path, [batch_path.relative_to(tmp_path)])

    df = clean(_base_config())

    assert len(df) == 1
    assert df.iloc[0]["species"] == "Gadus morhua"


def test_clean_drops_duplicate_species_sequence_pairs(project_dir):
    tmp_path, raw_dir = project_dir
    record = {
        "species": "Gadus morhua",
        "genus": "Gadus",
        "family": "Gadidae",
        "order": "Gadiformes",
        "nuc": GOOD_SEQ,
    }
    batch_path = _write_batch(raw_dir, "batch_0000.json", [record, dict(record)])
    _write_manifest(tmp_path, [batch_path.relative_to(tmp_path)])

    df = clean(_base_config())

    assert len(df) == 1


def test_clean_filters_by_length_and_ambiguous_bases(project_dir):
    tmp_path, raw_dir = project_dir
    too_short = "ACGT" * 10  # 40bp
    too_ambiguous = ("N" * 60) + ("ACGT" * 135)  # 600bp, >1% N
    records = [
        {"species": "A a", "genus": "A", "family": "Fa", "order": "Oa", "nuc": GOOD_SEQ},
        {"species": "B b", "genus": "B", "family": "Fb", "order": "Ob", "nuc": too_short},
        {"species": "C c", "genus": "C", "family": "Fc", "order": "Oc", "nuc": too_ambiguous},
    ]
    batch_path = _write_batch(raw_dir, "batch_0000.json", records)
    _write_manifest(tmp_path, [batch_path.relative_to(tmp_path)])

    df = clean(_base_config())

    assert len(df) == 1
    assert df.iloc[0]["species"] == "A a"


def test_clean_writes_parquet_with_expected_columns(project_dir):
    tmp_path, raw_dir = project_dir
    record = {
        "species": "Gadus morhua",
        "genus": "Gadus",
        "family": "Gadidae",
        "order": "Gadiformes",
        "nuc": GOOD_SEQ,
        "coord": [60.1, 5.3],  # BOLD returns [lat, lon] as a single array field
        "processid": "P1",
        "bin_uri": "BOLD:AAA0001",
    }
    batch_path = _write_batch(raw_dir, "batch_0000.json", [record])
    _write_manifest(tmp_path, [batch_path.relative_to(tmp_path)])

    clean(_base_config())

    output_path = tmp_path / "data" / "processed" / "sequences.parquet"
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert list(df.columns) == [
        "species",
        "genus",
        "family",
        "order",
        "sequence",
        "lat",
        "lon",
        "process_id",
        "bin_uri",
    ]
    assert df.iloc[0]["lat"] == 60.1
    assert df.iloc[0]["lon"] == 5.3


def test_clean_raises_if_manifest_missing(project_dir):
    with pytest.raises(FileNotFoundError):
        clean(_base_config())
