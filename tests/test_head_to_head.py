"""Offline unit tests for src.eval.head_to_head metrics, table, and plot."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
import yaml

from src.eval.head_to_head import (
    accuracy,
    build_table_rows,
    evaluate_fitted_split,
    false_confident_wrong_rate,
    fcw_curve,
    fit_predict_systems,
    format_markdown_table,
    plot_fcw_vs_threshold,
    run,
)


@dataclass
class _FakePrediction:
    species: str | None
    genus: str | None = None
    family: str | None = None
    order: str | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    nearest: dict[str, str] = field(default_factory=dict)


def test_accuracy_counts_abstention_as_incorrect():
    y_true = ["A", "A", "B"]
    y_pred = ["A", None, "WRONG"]
    assert accuracy(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_accuracy_empty_is_nan():
    assert np.isnan(accuracy([], []))


def test_false_confident_wrong_rate_only_counts_calls_above_threshold():
    y_true = ["A", "A", "A", "A"]
    y_pred = ["WRONG", "WRONG", None, "A"]
    scores = [0.9, 0.2, 0.99, 0.95]
    # Only the first row is a wrong call above 0.5.
    assert false_confident_wrong_rate(y_true, y_pred, scores, 0.5) == pytest.approx(0.25)
    # At 0.0 the low-confidence wrong call also counts (2/4); abstention never does.
    assert false_confident_wrong_rate(y_true, y_pred, scores, 0.0) == pytest.approx(0.5)


def test_fcw_curve_is_nonincreasing_as_threshold_rises():
    y_true = ["A"] * 10
    y_pred = ["WRONG"] * 10
    scores = [i / 10.0 for i in range(10)]
    curve = fcw_curve(y_true, y_pred, scores, [0.0, 0.5, 0.9, 1.0])
    assert curve == sorted(curve, reverse=True)


def test_build_table_is_method_by_rank_with_all_and_novel_metrics():
    result = {
        "systems": {
            "ours": {
                "all": {
                    rank: {"accuracy": 0.5, "false_confident_wrong_rate": 0.1}
                    for rank in ("species", "genus", "family", "order")
                },
                "novel": {
                    rank: {"accuracy": 0.0, "false_confident_wrong_rate": 0.0}
                    for rank in ("species", "genus", "family", "order")
                },
            },
            "blast": None,
            "naive_bayes": {
                "all": {
                    rank: {"accuracy": 0.8, "false_confident_wrong_rate": 0.2}
                    for rank in ("species", "genus", "family", "order")
                },
                "novel": {
                    rank: {"accuracy": 0.0, "false_confident_wrong_rate": 0.9}
                    for rank in ("species", "genus", "family", "order")
                },
            },
            "nearest_neighbor": {
                "all": {
                    rank: {"accuracy": 0.7, "false_confident_wrong_rate": 0.15}
                    for rank in ("species", "genus", "family", "order")
                },
                "novel": {
                    rank: {"accuracy": 0.0, "false_confident_wrong_rate": 0.85}
                    for rank in ("species", "genus", "family", "order")
                },
            },
        }
    }

    rows = build_table_rows(result)
    assert len(rows) == 16  # 4 methods x 4 ranks
    ours_species = next(r for r in rows if r["method"] == "ours" and r["rank"] == "species")
    assert ours_species["accuracy"] == 0.5
    assert ours_species["accuracy_novel"] == 0.0
    blast_species = next(r for r in rows if r["method"] == "blast" and r["rank"] == "species")
    assert blast_species["accuracy"] is None

    markdown = format_markdown_table(rows)
    assert "accuracy_novel" in markdown
    assert "false_confident_wrong_rate_novel" in markdown


def test_plot_writes_png(tmp_path):
    thresholds = [0.0, 0.5, 1.0]
    result = {
        "holdout_fraction": 0.5,
        "thresholds": thresholds,
        "systems": {
            "ours": {
                "novel": {
                    "species": {"fcw_curve": [0.2, 0.05, 0.0]},
                }
            },
            "blast": {
                "novel": {
                    "species": {"fcw_curve": [0.4, 0.3, 0.1]},
                }
            },
            "naive_bayes": {
                "novel": {
                    "species": {"fcw_curve": [0.9, 0.8, 0.1]},
                }
            },
            "nearest_neighbor": {
                "novel": {
                    "species": {"fcw_curve": [0.85, 0.6, 0.05]},
                }
            },
        },
    }
    out = tmp_path / "fcw_vs_threshold.png"
    plot_fcw_vs_threshold(result, out, rank="species", subset="novel")
    assert out.exists()
    assert out.stat().st_size > 0


def test_evaluate_fitted_split_separates_all_vs_novel():
    val_df = pd.DataFrame({"species": ["A"], "genus": ["GA"], "family": ["FA"], "order": ["OA"]})
    test_df = pd.DataFrame({"species": ["B"], "genus": ["GB"], "family": ["FB"], "order": ["OB"]})
    val_pred = [
        _FakePrediction(
            species="A",
            genus="GA",
            family="FA",
            order="OA",
            confidence={rank: 0.9 for rank in ("species", "genus", "family", "order")},
            nearest={"species": "A", "genus": "GA", "family": "FA", "order": "OA"},
        )
    ]
    test_pred = [
        _FakePrediction(
            species=None,
            genus=None,
            family="FB",
            order="OB",
            confidence={"species": 0.1, "genus": 0.2, "family": 0.8, "order": 0.9},
            nearest={"species": "A", "genus": "GA", "family": "FB", "order": "OB"},
        )
    ]
    fitted = {
        "holdout_fraction": 0.5,
        "n_genera_held_out": 1,
        "n_genera_total": 2,
        "val_df": val_df,
        "test_df": test_df,
        "predictions": {
            "ours": {"val": val_pred, "test": test_pred},
            "blast": {"val": None, "test": None},
            "naive_bayes": {"val": val_pred, "test": test_pred},
            "nearest_neighbor": {"val": val_pred, "test": test_pred},
        },
    }
    result = evaluate_fitted_split(fitted, threshold=0.5, thresholds=[0.0, 0.5, 1.0], plot_rank="species")
    ours_all = result["systems"]["ours"]["all"]["species"]
    ours_novel = result["systems"]["ours"]["novel"]["species"]
    assert ours_all["n"] == 2
    assert ours_novel["n"] == 1
    # Novel species call abstained → accuracy 0, FCW 0 at the operating point.
    assert ours_novel["accuracy"] == 0.0
    assert ours_novel["false_confident_wrong_rate"] == 0.0
    # Sweep uses nearest label (always wrong at species on novel) with low confidence.
    assert ours_novel["fcw_curve"][0] == 1.0  # threshold 0.0
    assert ours_novel["fcw_curve"][1] == 0.0  # threshold 0.5 > 0.1 confidence
    assert result["systems"]["blast"] is None


def _random_sequence(seed: int, length: int = 80) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _synthetic_corpus() -> tuple[pd.DataFrame, dict]:
    rows = []
    # Two seen-genera species (train+val) and one held-out genus (test).
    specs = [
        ("SeenA a", "SeenA", "FamA", "OrdA", 0),
        ("SeenA b", "SeenA", "FamA", "OrdA", 10),
        ("SeenB a", "SeenB", "FamB", "OrdA", 20),
        ("NovelX x", "NovelX", "FamA", "OrdA", 90),
    ]
    for species, genus, family, order, seed_base in specs:
        for j in range(4):
            rows.append(
                {
                    "process_id": f"{species.replace(' ', '_')}_{j}",
                    "species": species,
                    "genus": genus,
                    "family": family,
                    "order": order,
                    "sequence": _random_sequence(seed_base + j),
                    "lat": float(seed_base),
                    "lon": float(j),
                }
            )
    df = pd.DataFrame(rows)
    train_ids = df[df["genus"] != "NovelX"].groupby("species").head(3)["process_id"].tolist()
    val_ids = df[df["genus"] != "NovelX"].groupby("species").tail(1)["process_id"].tolist()
    test_ids = df[df["genus"] == "NovelX"]["process_id"].tolist()
    split = {
        "holdout_fraction": 0.5,
        "n_genera_held_out": 1,
        "n_genera_total": 3,
        "train_process_ids": train_ids,
        "val_process_ids": val_ids,
        "test_process_ids": test_ids,
    }
    return df, split


def _tiny_config(tmp_path) -> dict:
    return {
        "encoder": {"k": 4, "n_components": 4, "random_state": 0},
        "classifier": {"seq_weight": 0.8, "geo_weight": 0.2},
        "fallback": {"rank_percentile": 90.0, "sample_count_prior": 5.0},
        "baselines": {
            "naive_bayes": {"k": 4},
            "blast": {"db_dir": str(tmp_path / "blast_db"), "evalue": 1.0, "min_pident": 97.0},
        },
        "head_to_head": {
            "default_confidence_threshold": 0.5,
            "blast_bitscore_offset": 50.0,
        },
    }


def test_fit_predict_systems_scores_nb_and_ours_on_synthetic_split(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    df, split = _synthetic_corpus()
    config = _tiny_config(tmp_path)

    fitted = fit_predict_systems(split, df, config)
    result = evaluate_fitted_split(
        fitted,
        threshold=0.5,
        thresholds=[0.0, 0.5, 1.0],
        plot_rank="species",
    )

    nb_novel = result["systems"]["naive_bayes"]["novel"]["species"]
    ours_novel = result["systems"]["ours"]["novel"]["species"]
    # NB always answers; held-out genus cannot be a correct species call.
    assert nb_novel["n_called"] == nb_novel["n"]
    assert nb_novel["accuracy"] == 0.0
    # Our fallback should not confidently mis-call species on a novel genus.
    assert ours_novel["false_confident_wrong_rate"] <= nb_novel["false_confident_wrong_rate"]
    assert ours_novel["n_called"] < nb_novel["n_called"]


def test_run_writes_table_and_plot(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    df, split = _synthetic_corpus()
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)
    df.to_parquet(data_dir / "processed" / "sequences.parquet")
    (data_dir / "splits" / "holdout_50.json").write_text(
        __import__("json").dumps(split),
        encoding="utf-8",
    )

    config_path = tmp_path / "eval.yaml"
    config = {
        "data": {"sequences_path": "data/processed/sequences.parquet"},
        "splitter": {"holdout_fractions": [0.5], "splits_dir": "data/splits"},
        "encoder": {"k": 4, "n_components": 4, "random_state": 0},
        "classifier": {"seq_weight": 0.8, "geo_weight": 0.2},
        "fallback": {"rank_percentile": 90.0, "sample_count_prior": 5.0},
        "baselines": {
            "naive_bayes": {"k": 4},
            "blast": {"db_dir": "data/blast_db", "evalue": 1.0, "min_pident": 97.0},
        },
        "head_to_head": {
            "default_confidence_threshold": 0.5,
            "table_holdout_fraction": 0.5,
            "plot_rank": "species",
            "plot_subset": "novel",
            "n_thresholds": 5,
            "output_dir": "data/eval_results/head_to_head",
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    payload = run(yaml.safe_load(config_path.read_text(encoding="utf-8")))

    out_dir = tmp_path / "data" / "eval_results" / "head_to_head"
    assert (out_dir / "table.csv").exists()
    assert (out_dir / "table.md").exists()
    assert (out_dir / "fcw_vs_threshold.png").exists()
    assert (out_dir / "head_to_head.json").exists()
    assert payload["table"]
    methods = {row["method"] for row in payload["table"]}
    assert methods == {"ours", "blast", "naive_bayes", "nearest_neighbor"}
