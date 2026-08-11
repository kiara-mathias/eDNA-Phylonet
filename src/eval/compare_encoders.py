"""CLI: same fallback, old vs. new encoder (re-runs Step 1 + Step 2 metrics).

Usage::

    python -m src.eval.compare_encoders --config configs/eval.yaml

Fits each named encoder on the *same* clade-exclusion splits, then runs
the identical ``HierarchicalFallback`` (same seq/geo weights, percentile,
sample-count prior). Reports, per encoder:

- **Step 1**: pooled held-out ECE per rank (raw confidence).
- **Step 2**: on the novel/held-out-genera test set, accuracy (abstention
  counts as wrong) and false-confident-wrong-call rate at the configured
  threshold.

That contrast is the point: if novelty recall / family-resolved accuracy /
FCW stay in the same regime when k-mer+PCA is swapped for the CNN, the
fallback is encoder-agnostic.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402
from src.eval.calibration import evaluate_rank_calibration  # noqa: E402
from src.fallback.novelty import FallbackPrediction, HierarchicalFallback  # noqa: E402
from src.features.encoder import build_encoder, encoder_name  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def accuracy_abstain_wrong(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    correct = sum(pred is not None and pred == true for true, pred in zip(y_true, y_pred))
    return float(correct) / float(len(y_true))


def false_confident_wrong_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    count = 0
    for true, pred, score in zip(y_true, y_pred, scores):
        if pred is None:
            continue
        if pred != true and float(score) >= threshold:
            count += 1
    return float(count) / float(n)


def config_for_encoder(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Deep-copy config and pin ``encoder.name`` (other encoder knobs stay)."""
    cfg = copy.deepcopy(config)
    encoder_cfg = dict(cfg.get("encoder") or {})
    encoder_cfg["name"] = name
    cfg["encoder"] = encoder_cfg
    return cfg


def fit_fallback_on_split(
    split: dict[str, Any],
    sequences_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]

    by_id = sequences_df.set_index("process_id")
    train_df = by_id.loc[split["train_process_ids"]].reset_index()
    val_df = by_id.loc[split["val_process_ids"]].reset_index()
    test_df = by_id.loc[split["test_process_ids"]].reset_index()

    encoder = build_encoder(encoder_cfg)
    train_embeddings = encoder.fit_transform(train_df["sequence"], train_df["species"])
    val_embeddings = encoder.transform(val_df["sequence"])
    test_embeddings = encoder.transform(test_df["sequence"])

    fallback = HierarchicalFallback(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
        rank_percentile=fallback_cfg["rank_percentile"],
        sample_count_prior=float(fallback_cfg.get("sample_count_prior", 5.0)),
    )
    fallback.fit(train_embeddings, train_df)
    fallback.calibrate(val_embeddings, val_df, _latlon_array(val_df))

    val_predictions = fallback.predict(val_embeddings, _latlon_array(val_df))
    test_predictions = fallback.predict(test_embeddings, _latlon_array(test_df))
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "encoder": encoder_name(encoder_cfg),
        "embedding_dim": int(train_embeddings.shape[1]),
    }


def _score_predictions(
    predictions: list[FallbackPrediction],
    truth_df: pd.DataFrame,
    *,
    n_bins: int,
    fcw_threshold: float,
) -> dict[str, Any]:
    novelty_recall = float(np.mean([p.is_novel for p in predictions])) if predictions else float("nan")
    abstention = float(np.mean([p.predicted_rank is None for p in predictions])) if predictions else float("nan")

    per_rank: dict[str, Any] = {}
    for rank in _RANKS:
        y_true = truth_df[rank].to_numpy()
        y_call = np.array([getattr(p, rank) for p in predictions], dtype=object)
        scores = np.array([float(p.confidence[rank]) for p in predictions], dtype=np.float64)
        nearest = np.array([p.nearest[rank] for p in predictions], dtype=object)
        correct = (nearest == y_true).astype(np.float64)
        report = evaluate_rank_calibration(scores, correct, rank, n_bins=n_bins)
        family_resolved_idx = [i for i, p in enumerate(predictions) if p.predicted_rank == rank]
        if family_resolved_idx:
            resolved_acc = float(
                (y_true[family_resolved_idx] == y_call[family_resolved_idx]).mean()
            )
        else:
            resolved_acc = float("nan")
        per_rank[rank] = {
            "accuracy": _finite_or_none(accuracy_abstain_wrong(y_true, y_call)),
            "false_confident_wrong_rate": _finite_or_none(
                false_confident_wrong_rate(y_true, y_call, scores, fcw_threshold)
            ),
            "ece": _finite_or_none(report.ece),
            "resolved_count": len(family_resolved_idx),
            "resolved_accuracy": _finite_or_none(resolved_acc),
        }

    return {
        "n": len(predictions),
        "novelty_detection_recall": _finite_or_none(novelty_recall),
        "abstention_rate": _finite_or_none(abstention),
        "ranks": per_rank,
    }


def evaluate_encoder(
    name: str,
    split: dict[str, Any],
    sequences_df: pd.DataFrame,
    config: dict[str, Any],
    *,
    n_bins: int,
    fcw_threshold: float,
) -> dict[str, Any]:
    cfg = config_for_encoder(config, name)
    fitted = fit_fallback_on_split(split, sequences_df, cfg)
    return {
        "encoder": name,
        "embedding_dim": fitted["embedding_dim"],
        "holdout_fraction": split["holdout_fraction"],
        "n_train": len(fitted["train_df"]),
        "n_val": len(fitted["val_df"]),
        "n_test": len(fitted["test_df"]),
        "val": _score_predictions(fitted["val_predictions"], fitted["val_df"], n_bins=n_bins, fcw_threshold=fcw_threshold),
        "novel": _score_predictions(
            fitted["test_predictions"], fitted["test_df"], n_bins=n_bins, fcw_threshold=fcw_threshold
        ),
    }


def build_table_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for rank in _RANKS:
            novel = result["novel"]["ranks"][rank]
            rows.append(
                {
                    "encoder": result["encoder"],
                    "holdout_fraction": result["holdout_fraction"],
                    "rank": rank,
                    "ece_novel": novel["ece"],
                    "accuracy_novel": novel["accuracy"],
                    "false_confident_wrong_rate_novel": novel["false_confident_wrong_rate"],
                    "novelty_detection_recall": result["novel"]["novelty_detection_recall"],
                    "family_resolved_accuracy": result["novel"]["ranks"]["family"]["resolved_accuracy"],
                }
            )
    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "encoder",
        "holdout",
        "rank",
        "ece_novel",
        "accuracy_novel",
        "fcw_novel",
        "novelty_recall",
        "family_resolved_acc",
    ]

    def _fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [
            row["encoder"],
            f"{row['holdout_fraction']:.0%}",
            row["rank"],
            row["ece_novel"],
            row["accuracy_novel"],
            row["false_confident_wrong_rate_novel"],
            row["novelty_detection_recall"],
            row["family_resolved_accuracy"],
        ]
        lines.append("| " + " | ".join(_fmt(v) for v in values) + " |")
    return "\n".join(lines) + "\n"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    splitter_cfg = config["splitter"]
    calib_cfg = config.get("calibration", {})
    comparison_cfg = config.get("encoder_comparison", {})
    n_bins = int(calib_cfg.get("n_bins", 10))
    fcw_threshold = float(comparison_cfg.get("default_confidence_threshold", 0.5))
    names = list(comparison_cfg.get("names") or ["kmer_pca", "conv_triplet"])
    output_dir = resolve_path(comparison_cfg.get("output_dir", "data/eval_results/encoder_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences_df = pd.read_parquet(resolve_path(data_cfg["sequences_path"]))
    splits_dir = resolve_path(splitter_cfg["splits_dir"])

    results: list[dict[str, Any]] = []
    for fraction in splitter_cfg["holdout_fractions"]:
        pct = round(fraction * 100)
        split_path = splits_dir / f"holdout_{pct}.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"No split file at {split_path}. Run `python -m src.eval.splitter --config <config>` first."
            )
        with open(split_path, "r", encoding="utf-8") as fh:
            split = json.load(fh)

        for name in names:
            logger.info("Encoder %s on holdout_%d ...", name, pct)
            result = evaluate_encoder(
                name, split, sequences_df, config, n_bins=n_bins, fcw_threshold=fcw_threshold
            )
            results.append(result)
            novel = result["novel"]
            logger.info(
                "  %s novelty_recall=%.3f family_resolved_acc=%s species_fcw=%s species_ece=%s",
                name,
                novel["novelty_detection_recall"] or float("nan"),
                novel["ranks"]["family"]["resolved_accuracy"],
                novel["ranks"]["species"]["false_confident_wrong_rate"],
                novel["ranks"]["species"]["ece"],
            )

    rows = build_table_rows(results)
    table_md = format_markdown_table(rows)
    for line in table_md.splitlines():
        logger.info("%s", line)

    payload = {
        "encoders": names,
        "fcw_threshold": fcw_threshold,
        "n_bins": n_bins,
        "note": "Identical HierarchicalFallback hyperparameters; only the embedding changes.",
        "table": rows,
        "splits": results,
    }
    json_path = output_dir / "encoder_comparison.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)
    (output_dir / "table.md").write_text(table_md, encoding="utf-8")
    logger.info("Wrote %s", json_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
