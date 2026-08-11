"""Offline Streamlit smoke tests for app/dashboard.py, using Streamlit's
own ``AppTest`` framework rather than a real browser -- this actually
drives the app's widget state correctly (a real browser session found via
generic DOM automation can desync from Streamlit's custom text_area
component's internal state), and needs no network or running server.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.dashboard import nearest_relatives_dataframe
from src.fallback.novelty import FallbackPrediction, NeighborHit

_DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def _random_sequence(seed: int, length: int = 80) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _synthetic_sequences_df(n_species: int = 8, n_per_species: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_species):
        for j in range(n_per_species):
            rows.append(
                {
                    "process_id": f"P{i}_{j}",
                    "species": f"Genus{i // 2} species{i}",
                    "genus": f"Genus{i // 2}",
                    "family": f"Family{i // 4}",
                    "order": "OrderA",
                    "sequence": _random_sequence(seed=i * 100 + j),
                    "lat": 10.0 + i,
                    "lon": 20.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _write_app_config(root: Path) -> None:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "app.yaml").write_text(
        """
data:
  sequences_path: data/processed/sequences.parquet
val_fraction: 0.25
seed: 42
encoder:
  k: 3
  n_components: 5
  random_state: 42
classifier:
  seq_weight: 0.8
  geo_weight: 0.2
fallback:
  rank_percentile: 90.0
benchmark_results_path: data/eval_results/benchmark.json
calibration_results_path: data/eval_results/calibration/calibration.json
head_to_head_results_path: data/eval_results/head_to_head/head_to_head.json
""",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """A minimal, self-contained repo root with its own app.yaml + data.

    Clears Streamlit's ``st.cache_resource`` cache -- otherwise
    ``_cached_pipeline`` (keyed on the literal string "configs/app.yaml",
    identical across tests) would return a previous test's cached
    ``Pipeline`` (or ``None``) instead of rebuilding it against this test's
    isolated data.
    """
    st.cache_resource.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    _write_app_config(tmp_path)
    return tmp_path


def test_dashboard_shows_missing_data_warning_when_no_sequences_file(isolated_project):
    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[0].warning) >= 1


def test_dashboard_classifies_a_known_sequence(isolated_project):
    df = _synthetic_sequences_df()
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    df.to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.tabs[0].text_area[0].set_value(df.iloc[0]["sequence"])
    at.tabs[0].button[0].click().run(timeout=60)

    assert not at.exception
    assert len(at.tabs[0].success) + len(at.tabs[0].warning) >= 1


def test_dashboard_rejects_invalid_sequence_text(isolated_project):
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    _synthetic_sequences_df().to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    at.tabs[0].text_area[0].set_value("not a real sequence !!!")
    at.tabs[0].button[0].click().run(timeout=60)

    assert not at.exception
    assert any("A/C/G/T/N" in e.value for e in at.tabs[0].error)


def test_dashboard_renders_benchmark_tab_when_results_exist(isolated_project):
    results_dir = isolated_project / "data" / "eval_results"
    results_dir.mkdir(parents=True)
    results = [
        {
            "holdout_fraction": 0.3,
            "n_genera_held_out": 45,
            "n_genera_total": 151,
            "n_train": 100,
            "n_val": 20,
            "n_test": 50,
            "systems": {
                "ours": {
                    rank: {"coverage": 0.5, "accuracy_among_answered": 0.8}
                    for rank in ("species", "genus", "family", "order")
                },
                "blast": None,
            },
        }
    ]
    (results_dir / "benchmark.json").write_text(json.dumps(results), encoding="utf-8")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[1].dataframe) >= 1


def test_nearest_relatives_dataframe_lists_species_then_genera_in_given_order():
    prediction = FallbackPrediction(
        predicted_rank=None,
        is_novel=True,
        closest_relative_species="A a",
        species=None,
        genus=None,
        family=None,
        order=None,
        confidence={rank: 0.0 for rank in ("species", "genus", "family", "order")},
        distance={rank: 1.0 for rank in ("species", "genus", "family", "order")},
        support={rank: 1 for rank in ("species", "genus", "family", "order")},
        nearest={rank: "A a" if rank == "species" else "A" for rank in ("species", "genus", "family", "order")},
        nearest_species=[
            NeighborHit(label="A a", rank="species", distance=0.1, support=3, genus="A", family="FA", order="OA"),
            NeighborHit(label="B b", rank="species", distance=0.4, support=2, genus="B", family="FB", order="OB"),
        ],
        nearest_genera=[
            NeighborHit(label="A", rank="genus", distance=0.2, support=5, genus="A", family="FA", order="OA"),
        ],
    )

    table = nearest_relatives_dataframe(prediction)

    assert list(table["Label"]) == ["A a", "B b", "A"]
    assert list(table["Taxon rank"]) == ["Species", "Species", "Genus"]
    assert list(table["Rank"]) == [1, 2, 1]


def test_dashboard_gallery_tab_renders_hardcoded_examples(isolated_project):
    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    gallery = at.tabs[3]
    assert len(gallery.expander) == 5


def test_dashboard_calibration_tab_shows_missing_message(isolated_project):
    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[2].info) >= 1


def test_dashboard_calibration_tab_renders_saved_curve(isolated_project):
    calib_dir = isolated_project / "data" / "eval_results" / "calibration"
    calib_dir.mkdir(parents=True)
    summary = {
        "n_bins": 2,
        "decision": "pass",
        "raw": {
            "ece_by_rank": {"species": 0.05, "genus": 0.04, "family": 0.08, "order": 0.02},
            "reports": [
                {
                    "rank": rank,
                    "n": 10,
                    "ece": 0.05,
                    "bins": [
                        {
                            "bin_index": 0,
                            "conf_low": 0.0,
                            "conf_high": 0.5,
                            "mean_confidence": 0.25,
                            "accuracy": 0.3,
                            "count": 4,
                        },
                        {
                            "bin_index": 1,
                            "conf_low": 0.5,
                            "conf_high": 1.0,
                            "mean_confidence": 0.8,
                            "accuracy": 0.7,
                            "count": 6,
                        },
                    ],
                }
                for rank in ("species", "genus", "family", "order")
            ],
        },
        "recalibrated": None,
    }
    (calib_dir / "calibration.json").write_text(json.dumps(summary), encoding="utf-8")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[2].metric) >= 4


def test_dashboard_fcw_tab_renders_when_results_exist(isolated_project):
    out_dir = isolated_project / "data" / "eval_results" / "head_to_head"
    out_dir.mkdir(parents=True)
    payload = {
        "table_holdout_fraction": 0.5,
        "default_confidence_threshold": 0.5,
        "splits": [
            {
                "holdout_fraction": 0.5,
                "thresholds": [0.0, 0.5, 1.0],
                "systems": {
                    "ours": {
                        "novel": {
                            rank: {"fcw_curve": [0.2, 0.1, 0.0]}
                            for rank in ("species", "genus", "family", "order")
                        },
                        "all": {
                            rank: {"fcw_curve": [0.3, 0.15, 0.0]}
                            for rank in ("species", "genus", "family", "order")
                        },
                    },
                    "blast": None,
                },
            }
        ],
    }
    (out_dir / "head_to_head.json").write_text(json.dumps(payload), encoding="utf-8")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[4].slider) >= 1
    assert len(at.tabs[4].multiselect) >= 1


def test_dashboard_shows_missing_benchmark_message_when_no_results(isolated_project):
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    _synthetic_sequences_df().to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.tabs[1].info) >= 1
