"""Offline tests for same-fallback encoder comparison."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import yaml

from src.eval.compare_encoders import (
    accuracy_abstain_wrong,
    config_for_encoder,
    evaluate_encoder,
    false_confident_wrong_rate,
    format_markdown_table,
    run,
)
from src.eval.splitter import make_split
from src.features.dnabert import HuggingFaceDNAEncoder


def _random_sequence(seed: int, length: int = 48) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _synthetic_corpus(n_genera: int = 6, species_per_genus: int = 2, rows_per_species: int = 8) -> pd.DataFrame:
    rows = []
    pid = 0
    for g in range(n_genera):
        for s in range(species_per_genus):
            species = f"Genus{g} species{s}"
            for r in range(rows_per_species):
                rows.append(
                    {
                        "process_id": f"P{pid}",
                        "species": species,
                        "genus": f"Genus{g}",
                        "family": f"Family{g // 2}",
                        "order": "OrderA",
                        "sequence": _random_sequence(seed=pid * 17 + r, length=48),
                        "lat": float(g),
                        "lon": float(s),
                    }
                )
                pid += 1
    return pd.DataFrame(rows)


def _tiny_config() -> dict:
    return {
        "encoder": {
            "name": "kmer_pca",
            "k": 3,
            "n_components": 4,
            "random_state": 0,
            "conv_triplet": {
                "embedding_dim": 6,
                "n_filters": 4,
                "kernel_size": 3,
                "max_len": 48,
                "epochs": 2,
                "batch_size": 8,
                "learning_rate": 0.05,
                "margin": 0.4,
                "max_train_sequences": 80,
                "random_state": 0,
            },
        },
        "classifier": {"seq_weight": 0.8, "geo_weight": 0.2},
        "fallback": {"rank_percentile": 90.0, "sample_count_prior": 5.0},
        "calibration": {"n_bins": 5},
        "encoder_comparison": {
            "names": ["kmer_pca", "conv_triplet"],
            "default_confidence_threshold": 0.5,
            "output_dir": "data/eval_results/encoder_comparison",
        },
    }


def test_accuracy_counts_abstention_as_wrong():
    y_true = np.array(["A", "A", "B"])
    y_pred = np.array(["A", None, "WRONG"], dtype=object)
    assert accuracy_abstain_wrong(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_false_confident_wrong_ignores_abstention():
    y_true = np.array(["A", "A", "A"])
    y_pred = np.array(["WRONG", None, "A"], dtype=object)
    scores = np.array([0.9, 0.99, 0.95])
    assert false_confident_wrong_rate(y_true, y_pred, scores, 0.5) == pytest.approx(1.0 / 3.0)


def test_config_for_encoder_pins_name_without_dropping_knn_keys():
    cfg = config_for_encoder(_tiny_config(), "conv_triplet")
    assert cfg["encoder"]["name"] == "conv_triplet"
    assert cfg["encoder"]["k"] == 3
    assert cfg["encoder"]["conv_triplet"]["embedding_dim"] == 6


def test_evaluate_encoder_kmer_and_cnn_share_fallback_shape():
    df = _synthetic_corpus()
    split = make_split(df, holdout_fraction=0.3, seed=0, val_fraction=0.25)
    config = _tiny_config()

    kmer = evaluate_encoder("kmer_pca", split, df, config, n_bins=5, fcw_threshold=0.5)
    cnn = evaluate_encoder("conv_triplet", split, df, config, n_bins=5, fcw_threshold=0.5)

    assert kmer["encoder"] == "kmer_pca"
    assert cnn["encoder"] == "conv_triplet"
    assert kmer["n_test"] == cnn["n_test"] == len(split["test_process_ids"])
    for result in (kmer, cnn):
        assert 0.0 <= result["novel"]["novelty_detection_recall"] <= 1.0
        assert "species" in result["novel"]["ranks"]
        assert result["novel"]["ranks"]["species"]["ece"] is None or result["novel"]["ranks"]["species"]["ece"] >= 0.0


def test_run_writes_comparison_table(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    df = _synthetic_corpus()
    split = make_split(df, holdout_fraction=0.5, seed=0, val_fraction=0.25)
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "splits").mkdir(parents=True)
    df.to_parquet(data_dir / "processed" / "sequences.parquet")
    (data_dir / "splits" / "holdout_50.json").write_text(__import__("json").dumps(split), encoding="utf-8")

    config = _tiny_config()
    config["data"] = {"sequences_path": "data/processed/sequences.parquet"}
    config["splitter"] = {"holdout_fractions": [0.5], "splits_dir": "data/splits"}
    (tmp_path / "eval.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    payload = run(yaml.safe_load((tmp_path / "eval.yaml").read_text(encoding="utf-8")))

    out = tmp_path / "data" / "eval_results" / "encoder_comparison"
    assert (out / "encoder_comparison.json").exists()
    assert (out / "table.md").exists()
    encoders = {row["encoder"] for row in payload["table"]}
    assert encoders == {"kmer_pca", "conv_triplet"}
    md = format_markdown_table(payload["table"])
    assert "kmer_pca" in md and "conv_triplet" in md


def test_dnabert_fit_without_transformers_raises_import_error():
    encoder = HuggingFaceDNAEncoder()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="dnabert"):
            encoder.fit(["ACGT"])
        return
    pytest.skip("transformers/torch installed; skip the missing-dep assertion")
