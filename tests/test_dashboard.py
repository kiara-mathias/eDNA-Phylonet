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

from app.dashboard import (
    blast_lie_cases,
    build_depth_tree_html,
    format_specimen_sequence,
    rank_visual_state,
    reliability_reports,
    species_fcw_methods,
)
from src.fallback.novelty import FallbackPrediction

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
calibration_summary_path: data/eval_results/calibration/calibration.json
calibration_predictions_path: data/eval_results/calibration/heldout_predictions.parquet
""",
        encoding="utf-8",
    )


def _prediction(**overrides) -> FallbackPrediction:
    base = dict(
        predicted_rank="genus",
        is_novel=True,
        closest_relative_species="Gadus morhua",
        species=None,
        genus="Gadus",
        family="Gadidae",
        order="Gadiformes",
        confidence={"species": 0.12, "genus": 0.71, "family": 0.80, "order": 0.90},
        distance={"species": 1.2, "genus": 0.4, "family": 0.3, "order": 0.2},
        support={"species": 3, "genus": 12, "family": 40, "order": 80},
        nearest={"species": "Gadus morhua", "genus": "Gadus", "family": "Gadidae", "order": "Gadiformes"},
    )
    base.update(overrides)
    return FallbackPrediction(**base)


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
    assert len(at.warning) >= 1


def test_dashboard_classifies_a_known_sequence(isolated_project):
    df = _synthetic_sequences_df()
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    df.to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)
    assert not at.exception

    at.text_area[0].set_value(df.iloc[0]["sequence"])
    at.button[0].click().run(timeout=60)

    assert not at.exception
    assert len(at.success) + len(at.warning) >= 1


def test_dashboard_rejects_invalid_sequence_text(isolated_project):
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    _synthetic_sequences_df().to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    at.text_area[0].set_value("not a real sequence !!!")
    at.button[0].click().run(timeout=60)

    assert not at.exception
    assert any("A/C/G/T/N" in e.value for e in at.error)


def test_dashboard_renders_benchmark_table_when_results_exist(isolated_project):
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
    assert len(at.dataframe) >= 1


def test_dashboard_omits_benchmark_when_no_results(isolated_project):
    data_dir = isolated_project / "data" / "processed"
    data_dir.mkdir(parents=True)
    _synthetic_sequences_df().to_parquet(data_dir / "sequences.parquet")

    at = AppTest.from_file(_DASHBOARD_PATH)
    at.run(timeout=60)

    assert not at.exception
    assert len(at.dataframe) == 0


def test_rank_visual_state_marks_shallower_ranks_skipped():
    prediction = _prediction(predicted_rank="genus")
    assert rank_visual_state(prediction, "species") == "skipped"
    assert rank_visual_state(prediction, "genus") == "committed"
    assert rank_visual_state(prediction, "family") == "abyss"
    assert rank_visual_state(prediction, "order") == "abyss"


def test_depth_tree_html_includes_taxon_and_reveal_delays():
    html = build_depth_tree_html(_prediction())
    assert "Gadus" in html
    assert "depth-rank" in html
    assert "--delay:0ms" in html
    assert "is-committed" in html
    assert "is-skipped" in html


def test_format_specimen_sequence_groups_bases():
    assert format_specimen_sequence("ACGTACGTAC", width=80) == "ACGTACGTAC"


def test_blast_lie_cases_picks_confident_wrong_novel_queries():
    df = pd.DataFrame(
        [
            {
                "split": "test",
                "is_novel": True,
                "nearest_species": "Wrongus wrongus",
                "true_species": "Truus truus",
                "confidence_species": 0.91,
                "predicted_rank": "family",
            },
            {
                "split": "test",
                "is_novel": True,
                "nearest_species": "Truus truus",
                "true_species": "Truus truus",
                "confidence_species": 0.99,
                "predicted_rank": "species",
            },
            {
                "split": "val",
                "is_novel": False,
                "nearest_species": "Wrongus wrongus",
                "true_species": "Truus truus",
                "confidence_species": 0.95,
                "predicted_rank": "species",
            },
        ]
    )
    cases = blast_lie_cases(df, limit=4)
    assert len(cases) == 1
    assert cases.iloc[0]["nearest_species"] == "Wrongus wrongus"


def test_species_fcw_methods_abstain_when_not_committed():
    df = pd.DataFrame(
        {
            "true_species": ["A", "B"],
            "nearest_species": ["A", "X"],
            "confidence_species": [0.9, 0.8],
            "predicted_rank": ["species", "family"],
        }
    )
    methods = species_fcw_methods(df)
    _, y_fallback, _ = methods["hierarchical fallback"]
    _, y_always, _ = methods["always-answer (species)"]
    assert y_fallback[0] == "A"
    assert y_fallback[1] is None
    assert list(y_always) == ["A", "X"]


def test_reliability_reports_prefers_recalibrated():
    summary = {
        "raw": {"reports": [{"rank": "species", "ece": 0.4, "bins": []}]},
        "recalibrated": {"reports": [{"rank": "species", "ece": 0.05, "bins": []}]},
    }
    reports = reliability_reports(summary)
    assert reports[0]["ece"] == 0.05
