"""CLI: validate the Step 5 hierarchical fallback / novelty-flagging layer
against each saved clade-exclusion split.

Usage::

    python -m src.eval.validate_fallback --config configs/eval.yaml

For each split, fits the Step 4 encoder + ``HierarchicalFallback`` on
train, calibrates confidence thresholds on val (in-distribution, seen
species), then reports:

- **Novelty-detection recall** on test (held-out-genus queries, which are
  genuinely novel at species/genus level): the fraction correctly flagged
  ``is_novel=True``.
- **False-novelty-flag rate** on val (in-distribution, seen queries): the
  fraction incorrectly *not* confidently classified at species level -- a
  calibration sanity check, expected near ``100 - rank_percentile``%.
- **Abstention rate** on test: the fraction fully unresolved
  (``predicted_rank is None``, i.e. not even confident at order level).
- **Resolved-rank accuracy** on test: among predictions that *did* resolve
  at a given rank, the fraction whose predicted label at that rank matches
  the truth -- answers "when the fallback commits to a rank, is it right?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/eval/validate_fallback.py`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402
from src.fallback.novelty import HierarchicalFallback  # noqa: E402
from src.features.kmer_pca import KmerPCAEncoder  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def evaluate_split(split: dict[str, Any], sequences_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]

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

    fallback = HierarchicalFallback(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
        rank_percentile=fallback_cfg["rank_percentile"],
        sample_count_prior=float(fallback_cfg.get("sample_count_prior", 5.0)),
        n_nearest_relatives=int(fallback_cfg.get("n_nearest_relatives", 5)),
    )
    fallback.fit(train_embeddings, train_df)
    fallback.calibrate(val_embeddings, val_df, _latlon_array(val_df))

    test_predictions = fallback.predict(test_embeddings, _latlon_array(test_df))
    val_predictions = fallback.predict(val_embeddings, _latlon_array(val_df))

    novelty_detection_recall = float(np.mean([p.is_novel for p in test_predictions])) if test_predictions else float("nan")
    abstention_rate = float(np.mean([p.predicted_rank is None for p in test_predictions])) if test_predictions else float("nan")

    resolved_counts: dict[str, int] = {}
    resolved_accuracy: dict[str, float] = {}
    for rank in _RANKS:
        idx = [i for i, p in enumerate(test_predictions) if p.predicted_rank == rank]
        resolved_counts[rank] = len(idx)
        if idx:
            y_true = test_df[rank].to_numpy()[idx]
            y_pred = np.array([getattr(test_predictions[i], rank) for i in idx])
            resolved_accuracy[rank] = float((y_true == y_pred).mean())
        else:
            resolved_accuracy[rank] = float("nan")

    false_novelty_flag_rate = float(np.mean([p.is_novel for p in val_predictions])) if val_predictions else float("nan")
    confident_val_idx = [i for i, p in enumerate(val_predictions) if p.predicted_rank == "species"]
    if confident_val_idx:
        y_true = val_df["species"].to_numpy()[confident_val_idx]
        y_pred = np.array([val_predictions[i].species for i in confident_val_idx])
        species_accuracy_when_confident = float((y_true == y_pred).mean())
    else:
        species_accuracy_when_confident = float("nan")

    return {
        "holdout_fraction": split["holdout_fraction"],
        "n_genera_held_out": split["n_genera_held_out"],
        "n_genera_total": split["n_genera_total"],
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "thresholds": fallback.thresholds_,
        "test": {
            "novelty_detection_recall": novelty_detection_recall,
            "abstention_rate": abstention_rate,
            "resolved_counts": resolved_counts,
            "resolved_accuracy": resolved_accuracy,
        },
        "val": {
            "false_novelty_flag_rate": false_novelty_flag_rate,
            "species_accuracy_when_confident": species_accuracy_when_confident,
        },
    }


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

        logger.info(
            "Evaluating holdout_%d (%d/%d genera held out)...", pct, split["n_genera_held_out"], split["n_genera_total"]
        )
        result = evaluate_split(split, sequences_df, config)
        results.append(result)

        logger.info(
            "  n_train=%d n_val=%d n_test=%d | novelty_recall=%.3f abstention=%.3f | "
            "val_false_flag=%.3f val_species_acc_when_confident=%.3f",
            result["n_train"],
            result["n_val"],
            result["n_test"],
            result["test"]["novelty_detection_recall"],
            result["test"]["abstention_rate"],
            result["val"]["false_novelty_flag_rate"],
            result["val"]["species_accuracy_when_confident"],
        )
        logger.info(
            "  resolved (test): family=%d/%d acc=%.3f | order=%d/%d acc=%.3f",
            result["test"]["resolved_counts"]["family"],
            result["n_test"],
            result["test"]["resolved_accuracy"]["family"],
            result["test"]["resolved_counts"]["order"],
            result["n_test"],
            result["test"]["resolved_accuracy"]["order"],
        )

    out_path = resolve_path("data/eval_results/fallback.json")
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
