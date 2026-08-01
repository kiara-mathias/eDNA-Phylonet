"""CLI: validate the Step 4 base classifier (encoder + Paper2Classifier)
against each saved clade-exclusion split, standalone -- no fallback module
yet (that's Step 5).

Usage::

    python -m src.eval.validate_baseline --config configs/eval.yaml

For each split written by ``src/eval/splitter.py``, fits the encoder and
classifier on that split's train rows, predicts on its test rows (queries
from held-out genera), and reports per-rank (species/genus/family/order)
accuracy. Family/order accuracy on held-out genera is the key number: it
answers "when the exact species is unknown, does the system still land
close?" -- the same question the eventual BLAST/QIIME2 benchmark (Step 6)
will be compared against.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `python src/eval/validate_baseline.py`

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402
from src.features.kmer_pca import KmerPCAEncoder  # noqa: E402
from src.model.classifier import Paper2Classifier  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def evaluate_split(split: dict[str, Any], sequences_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Fit encoder+classifier on the split's train rows, evaluate on test rows."""
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]

    by_id = sequences_df.set_index("process_id")
    train_df = by_id.loc[split["train_process_ids"]].reset_index()
    test_df = by_id.loc[split["test_process_ids"]].reset_index()

    encoder = KmerPCAEncoder(
        k=encoder_cfg["k"],
        n_components=encoder_cfg["n_components"],
        random_state=encoder_cfg["random_state"],
    )
    train_embeddings = encoder.fit_transform(train_df["sequence"])
    test_embeddings = encoder.transform(test_df["sequence"])

    classifier = Paper2Classifier(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
    )
    classifier.fit(train_embeddings, train_df)

    predictions = classifier.predict(test_embeddings, _latlon_array(test_df))

    accuracy: dict[str, float] = {}
    for rank in _RANKS:
        y_true = test_df[rank].to_numpy()
        y_pred = np.array([getattr(p, rank) for p in predictions])
        accuracy[rank] = float((y_true == y_pred).mean()) if len(y_true) else float("nan")

    return {
        "holdout_fraction": split["holdout_fraction"],
        "n_genera_held_out": split["n_genera_held_out"],
        "n_genera_total": split["n_genera_total"],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "accuracy": accuracy,
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

        logger.info("Evaluating holdout_%d (%d/%d genera held out)...", pct, split["n_genera_held_out"], split["n_genera_total"])
        result = evaluate_split(split, sequences_df, config)
        results.append(result)

        acc = result["accuracy"]
        logger.info(
            "  n_train=%d n_test=%d | species=%.3f genus=%.3f family=%.3f order=%.3f",
            result["n_train"],
            result["n_test"],
            acc["species"],
            acc["genus"],
            acc["family"],
            acc["order"],
        )

    out_path = resolve_path("data/eval_results/baseline.json")
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
