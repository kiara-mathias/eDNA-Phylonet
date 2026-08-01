"""CLI: benchmark our system against the Step 6 baselines on each saved
clade-exclusion split.

Usage::

    python -m src.eval.benchmark --config configs/eval.yaml

For each split, fits our own encoder + ``Paper2Classifier`` +
``HierarchicalFallback`` pipeline (identical to ``validate_fallback.py``)
alongside three baselines -- ``NaiveBayesBaseline`` (QIIME2 stand-in),
``NearestNeighborBaseline`` (no-phylogeny ablation of our own encoder), and
``BlastBaseline`` (real BLAST+, skipped with a warning if ``blastn``/
``makeblastdb`` aren't on ``PATH``) -- and reports, per system and per rank:

- **coverage**: fraction of test queries given *any* answer at that rank.
- **accuracy_among_answered**: of those given an answer, fraction correct.

This project's core claim is that a graceful, calibrated fallback beats
both "fails outright" (BLAST) and "always overconfident" (the two ML
baselines) on genuinely novel (held-out-genus) queries -- this script is
what that claim is checked against.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/eval/benchmark.py`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import BaselinePrediction  # noqa: E402
from src.baselines.blast_baseline import BlastBaseline  # noqa: E402
from src.baselines.naive_bayes_baseline import NaiveBayesBaseline  # noqa: E402
from src.baselines.nearest_neighbor_baseline import NearestNeighborBaseline  # noqa: E402
from src.common import load_config, resolve_path  # noqa: E402
from src.fallback.novelty import FallbackPrediction, HierarchicalFallback  # noqa: E402
from src.features.kmer_pca import KmerPCAEncoder  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")
_Prediction = BaselinePrediction | FallbackPrediction


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def _score(predictions: Sequence[_Prediction], test_df: pd.DataFrame, rank: str) -> dict[str, float]:
    y_true = test_df[rank].to_numpy()
    y_pred = np.array([getattr(p, rank) for p in predictions], dtype=object)
    answered = np.array([v is not None for v in y_pred], dtype=bool)

    coverage = float(answered.mean()) if len(y_pred) else float("nan")
    if answered.any():
        accuracy = float((y_pred[answered] == y_true[answered]).mean())
    else:
        accuracy = float("nan")

    return {"coverage": coverage, "accuracy_among_answered": accuracy}


def _score_system(predictions: Sequence[_Prediction], test_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {rank: _score(predictions, test_df, rank) for rank in _RANKS}


def evaluate_split(split: dict[str, Any], sequences_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]
    baselines_cfg = config["baselines"]

    by_id = sequences_df.set_index("process_id")
    train_df = by_id.loc[split["train_process_ids"]].reset_index()
    val_df = by_id.loc[split["val_process_ids"]].reset_index()
    test_df = by_id.loc[split["test_process_ids"]].reset_index()

    encoder = KmerPCAEncoder(
        k=encoder_cfg["k"],
        n_components=encoder_cfg["n_components"],
        random_state=encoder_cfg["random_state"],
    )
    train_embeddings = encoder.fit_transform(train_df["sequence"])
    val_embeddings = encoder.transform(val_df["sequence"])
    test_embeddings = encoder.transform(test_df["sequence"])

    ours = HierarchicalFallback(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
        rank_percentile=fallback_cfg["rank_percentile"],
    )
    ours.fit(train_embeddings, train_df)
    ours.calibrate(val_embeddings, val_df, _latlon_array(val_df))
    ours_predictions = ours.predict(test_embeddings, _latlon_array(test_df))

    naive_bayes = NaiveBayesBaseline(k=baselines_cfg["naive_bayes"]["k"])
    naive_bayes.fit(train_df)
    naive_bayes_predictions = naive_bayes.predict(test_df)

    nearest_neighbor = NearestNeighborBaseline()
    nearest_neighbor.fit(train_embeddings, train_df)
    nearest_neighbor_predictions = nearest_neighbor.predict(test_embeddings)

    systems: dict[str, dict[str, dict[str, float]] | None] = {
        "ours": _score_system(ours_predictions, test_df),
        "naive_bayes": _score_system(naive_bayes_predictions, test_df),
        "nearest_neighbor": _score_system(nearest_neighbor_predictions, test_df),
    }

    if BlastBaseline.is_available():
        blast_cfg = baselines_cfg["blast"]
        pct = round(split["holdout_fraction"] * 100)
        db_dir = resolve_path(blast_cfg["db_dir"]) / f"holdout_{pct}"
        blast = BlastBaseline(db_dir=db_dir, evalue=blast_cfg["evalue"], min_pident=blast_cfg["min_pident"])
        blast.fit(train_df)
        blast_predictions = blast.predict(test_df)
        systems["blast"] = _score_system(blast_predictions, test_df)
    else:
        logger.warning(
            "blastn/makeblastdb not found on PATH -- skipping BLAST baseline for this split "
            "(install ncbi-blast+; already done in Dockerfile/CI)."
        )
        systems["blast"] = None

    return {
        "holdout_fraction": split["holdout_fraction"],
        "n_genera_held_out": split["n_genera_held_out"],
        "n_genera_total": split["n_genera_total"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "systems": systems,
    }


def _log_comparison_table(result: dict[str, Any]) -> None:
    logger.info(
        "holdout_%d (%d/%d genera held out, n_test=%d)",
        round(result["holdout_fraction"] * 100),
        result["n_genera_held_out"],
        result["n_genera_total"],
        result["n_test"],
    )
    for system_name, scores in result["systems"].items():
        if scores is None:
            logger.info("  %-17s skipped (unavailable)", system_name)
            continue
        parts = [
            f"{rank}=cov:{scores[rank]['coverage']:.2f}/acc:{scores[rank]['accuracy_among_answered']:.2f}"
            for rank in _RANKS
        ]
        logger.info("  %-17s %s", system_name, " ".join(parts))


def run(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    splitter_cfg = config["splitter"]

    sequences_df = pd.read_parquet(resolve_path(data_cfg["sequences_path"]))
    splits_dir = resolve_path(splitter_cfg["splits_dir"])

    results = []
    for fraction in splitter_cfg["holdout_fractions"]:
        pct = round(fraction * 100)
        split_path = splits_dir / f"holdout_{pct}.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"No split file at {split_path}. Run `python -m src.eval.splitter --config <config>` first."
            )
        with open(split_path, "r", encoding="utf-8") as fh:
            split = json.load(fh)

        result = evaluate_split(split, sequences_df, config)
        results.append(result)
        _log_comparison_table(result)

    out_path = resolve_path("data/eval_results/benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote %s", out_path)

    return results


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to the eval YAML config (default: configs/eval.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
