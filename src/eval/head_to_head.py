"""CLI: standardized head-to-head of BLAST, Naive Bayes, 1-NN, and our system.

Usage::

    python -m src.eval.head_to_head --config configs/eval.yaml

Fits every method on the *same* clade-exclusion train split, scores the
same val+test queries, and reports two metrics per method per rank:

- **accuracy**: fraction of queries whose operating-point call matches the
  true label. Abstention (``None``) counts as incorrect.
- **false-confident-wrong-call rate**: fraction of queries where the system
  *made a call*, the call is wrong, and the method's confidence/score is
  at or above a threshold. BLAST uses bitscore (falling back to e-value)
  as its confidence proxy.

Both metrics are also computed on the novel/held-out-genera subset (the
split's test set). That is the contrast this project's fallback is for:
BLAST has no hierarchical fallback, so on novel genera it either abstains
at every rank or copies a related species' full taxonomy from a high
bitscore hit.

Writes, under ``data/eval_results/head_to_head/``:

- ``table.csv`` / ``table.md`` -- one table (method x rank x metric)
- ``fcw_vs_threshold.png`` -- false-confident-wrong rate vs. threshold,
  one line per method (default: species rank, novel subset)
- ``head_to_head.json`` -- full per-split numbers and the sweep curves
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
_METHODS = ("ours", "blast", "naive_bayes", "nearest_neighbor")
_Prediction = BaselinePrediction | FallbackPrediction

_METHOD_LABELS = {
    "ours": "ours",
    "blast": "BLAST",
    "naive_bayes": "Naive Bayes",
    "nearest_neighbor": "1-NN",
}
_METHOD_COLORS = {
    "ours": "#1b7a3d",
    "blast": "#b3261e",
    "naive_bayes": "#2a6f97",
    "nearest_neighbor": "#c98a12",
}


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def _call_label(prediction: _Prediction, rank: str) -> str | None:
    return getattr(prediction, rank)


def _scored_label(prediction: _Prediction, rank: str) -> str | None:
    nearest = getattr(prediction, "nearest", None) or {}
    if rank in nearest:
        return nearest[rank]
    return _call_label(prediction, rank)


def _confidence(prediction: _Prediction, rank: str) -> float:
    conf = getattr(prediction, "confidence", None) or {}
    value = conf.get(rank, 0.0)
    return float(value) if value is not None and np.isfinite(value) else 0.0


def accuracy(y_true: Sequence[object], y_pred: Sequence[object | None]) -> float:
    """Fraction correct; unanswered (``None``) counts as incorrect."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    correct = sum(pred is not None and pred == true for true, pred in zip(y_true, y_pred))
    return float(correct) / float(n)


def false_confident_wrong_rate(
    y_true: Sequence[object],
    y_pred: Sequence[object | None],
    scores: Sequence[float],
    threshold: float,
) -> float:
    """Fraction of queries that are a wrong *call* with score >= threshold."""
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


def fcw_curve(
    y_true: Sequence[object],
    y_pred: Sequence[object | None],
    scores: Sequence[float],
    thresholds: Sequence[float],
) -> list[float]:
    return [false_confident_wrong_rate(y_true, y_pred, scores, t) for t in thresholds]


def _labels_and_scores(
    predictions: Sequence[_Prediction],
    truth_df: pd.DataFrame,
    rank: str,
    *,
    use_scored: bool,
) -> tuple[list[object], list[object | None], list[float]]:
    y_true = truth_df[rank].tolist()
    getter = _scored_label if use_scored else _call_label
    y_pred = [getter(p, rank) for p in predictions]
    scores = [_confidence(p, rank) for p in predictions]
    return y_true, y_pred, scores


def score_rank(
    predictions: Sequence[_Prediction],
    truth_df: pd.DataFrame,
    rank: str,
    threshold: float,
) -> dict[str, float]:
    y_true, y_call, call_scores = _labels_and_scores(predictions, truth_df, rank, use_scored=False)
    return {
        "accuracy": accuracy(y_true, y_call),
        "false_confident_wrong_rate": false_confident_wrong_rate(y_true, y_call, call_scores, threshold),
        "n": float(len(y_true)),
        "n_called": float(sum(p is not None for p in y_call)),
    }


def _default_head_to_head_cfg() -> dict[str, Any]:
    return {
        "default_confidence_threshold": 0.5,
        "table_holdout_fraction": 0.5,
        "plot_rank": "species",
        "plot_subset": "novel",
        "n_thresholds": 21,
        "output_dir": "data/eval_results/head_to_head",
        "blast_bitscore_offset": 50.0,
    }


def fit_predict_systems(
    split: dict[str, Any],
    sequences_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit all methods on ``split`` train; predict val (seen) and test (novel)."""
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]
    baselines_cfg = config["baselines"]
    h2h_cfg = {**_default_head_to_head_cfg(), **config.get("head_to_head", {})}

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
        sample_count_prior=float(fallback_cfg.get("sample_count_prior", 5.0)),
    )
    ours.fit(train_embeddings, train_df)
    ours.calibrate(val_embeddings, val_df, _latlon_array(val_df))

    naive_bayes = NaiveBayesBaseline(k=baselines_cfg["naive_bayes"]["k"])
    naive_bayes.fit(train_df)

    nearest_neighbor = NearestNeighborBaseline()
    nearest_neighbor.fit(train_embeddings, train_df)

    predictions: dict[str, dict[str, list[_Prediction] | None]] = {
        "ours": {
            "val": ours.predict(val_embeddings, _latlon_array(val_df)),
            "test": ours.predict(test_embeddings, _latlon_array(test_df)),
        },
        "naive_bayes": {
            "val": naive_bayes.predict(val_df),
            "test": naive_bayes.predict(test_df),
        },
        "nearest_neighbor": {
            "val": nearest_neighbor.predict(val_embeddings),
            "test": nearest_neighbor.predict(test_embeddings),
        },
        "blast": {"val": None, "test": None},
    }

    if BlastBaseline.is_available():
        blast_cfg = baselines_cfg["blast"]
        pct = round(split["holdout_fraction"] * 100)
        db_dir = resolve_path(blast_cfg["db_dir"]) / f"holdout_{pct}"
        blast = BlastBaseline(
            db_dir=db_dir,
            evalue=blast_cfg["evalue"],
            min_pident=blast_cfg["min_pident"],
            bitscore_offset=float(h2h_cfg["blast_bitscore_offset"]),
        )
        blast.fit(train_df)
        predictions["blast"] = {
            "val": blast.predict(val_df),
            "test": blast.predict(test_df),
        }
    else:
        logger.warning(
            "blastn/makeblastdb not found on PATH -- skipping BLAST for this split "
            "(install ncbi-blast+; already done in Dockerfile/CI)."
        )

    return {
        "holdout_fraction": split["holdout_fraction"],
        "n_genera_held_out": split["n_genera_held_out"],
        "n_genera_total": split["n_genera_total"],
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "predictions": predictions,
    }


def _concat_predictions(
    val_preds: Sequence[_Prediction] | None,
    test_preds: Sequence[_Prediction] | None,
) -> list[_Prediction] | None:
    if val_preds is None or test_preds is None:
        return None
    return list(val_preds) + list(test_preds)


def evaluate_fitted_split(
    fitted: dict[str, Any],
    *,
    threshold: float,
    thresholds: Sequence[float],
    plot_rank: str,
) -> dict[str, Any]:
    val_df: pd.DataFrame = fitted["val_df"]
    test_df: pd.DataFrame = fitted["test_df"]
    all_df = pd.concat([val_df, test_df], ignore_index=True)
    subsets = {
        "all": (all_df, "val+test (seen val + novel test)"),
        "novel": (test_df, "held-out genera (test)"),
    }

    systems: dict[str, Any] = {}
    for method in _METHODS:
        val_preds = fitted["predictions"][method]["val"]
        test_preds = fitted["predictions"][method]["test"]
        if val_preds is None or test_preds is None:
            systems[method] = None
            continue

        pred_by_subset = {
            "all": _concat_predictions(val_preds, test_preds),
            "novel": list(test_preds),
        }
        per_subset: dict[str, Any] = {}
        for subset_name, (truth_df, _desc) in subsets.items():
            preds = pred_by_subset[subset_name]
            assert preds is not None
            ranks: dict[str, Any] = {}
            for rank in _RANKS:
                metrics = score_rank(preds, truth_df, rank, threshold)
                y_true, y_scored, scores = _labels_and_scores(preds, truth_df, rank, use_scored=True)
                metrics["fcw_curve"] = fcw_curve(y_true, y_scored, scores, thresholds)
                ranks[rank] = metrics
            per_subset[subset_name] = ranks
        systems[method] = per_subset

    return {
        "holdout_fraction": fitted["holdout_fraction"],
        "n_genera_held_out": fitted["n_genera_held_out"],
        "n_genera_total": fitted["n_genera_total"],
        "n_val": len(val_df),
        "n_test": len(test_df),
        "threshold": float(threshold),
        "thresholds": [float(t) for t in thresholds],
        "plot_rank": plot_rank,
        "systems": systems,
    }


def build_table_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _METHODS:
        scores = result["systems"].get(method)
        if scores is None:
            for rank in _RANKS:
                rows.append(
                    {
                        "method": method,
                        "rank": rank,
                        "accuracy": None,
                        "false_confident_wrong_rate": None,
                        "accuracy_novel": None,
                        "false_confident_wrong_rate_novel": None,
                    }
                )
            continue
        for rank in _RANKS:
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "accuracy": scores["all"][rank]["accuracy"],
                    "false_confident_wrong_rate": scores["all"][rank]["false_confident_wrong_rate"],
                    "accuracy_novel": scores["novel"][rank]["accuracy"],
                    "false_confident_wrong_rate_novel": scores["novel"][rank]["false_confident_wrong_rate"],
                }
            )
    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "method",
        "rank",
        "accuracy",
        "false_confident_wrong_rate",
        "accuracy_novel",
        "false_confident_wrong_rate_novel",
    ]

    def _fmt(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def plot_fcw_vs_threshold(
    result: dict[str, Any],
    output_path: Path,
    *,
    rank: str,
    subset: str,
) -> None:
    import matplotlib.pyplot as plt

    thresholds = result["thresholds"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plotted = False
    for method in _METHODS:
        scores = result["systems"].get(method)
        if scores is None:
            continue
        ys = scores[subset][rank]["fcw_curve"]
        ax.plot(
            thresholds,
            ys,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=_METHOD_COLORS.get(method, "#333333"),
            label=_METHOD_LABELS.get(method, method),
        )
        plotted = True

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence / score threshold")
    ax.set_ylabel("False-confident-wrong-call rate")
    ax.set_title(
        f"False-confident wrong calls vs. threshold\n"
        f"({rank} rank, {subset} subset, holdout={result['holdout_fraction']:.0%})"
    )
    if plotted:
        ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _log_table(rows: list[dict[str, Any]], holdout_fraction: float) -> None:
    logger.info("Head-to-head table (holdout=%.0f%%)", holdout_fraction * 100)
    for line in format_markdown_table(rows).splitlines():
        logger.info("%s", line)


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
    h2h_cfg = {**_default_head_to_head_cfg(), **config.get("head_to_head", {})}

    sequences_df = pd.read_parquet(resolve_path(data_cfg["sequences_path"]))
    splits_dir = resolve_path(splitter_cfg["splits_dir"])
    threshold = float(h2h_cfg["default_confidence_threshold"])
    n_thresholds = int(h2h_cfg["n_thresholds"])
    thresholds = np.linspace(0.0, 1.0, n_thresholds).tolist()
    plot_rank = str(h2h_cfg["plot_rank"])
    plot_subset = str(h2h_cfg["plot_subset"])
    table_holdout = float(h2h_cfg["table_holdout_fraction"])

    results: list[dict[str, Any]] = []
    table_result: dict[str, Any] | None = None

    for fraction in splitter_cfg["holdout_fractions"]:
        pct = round(fraction * 100)
        split_path = splits_dir / f"holdout_{pct}.json"
        if not split_path.exists():
            raise FileNotFoundError(
                f"No split file at {split_path}. Run `python -m src.eval.splitter --config <config>` first."
            )
        with open(split_path, "r", encoding="utf-8") as fh:
            split = json.load(fh)

        fitted = fit_predict_systems(split, sequences_df, config)
        result = evaluate_fitted_split(
            fitted,
            threshold=threshold,
            thresholds=thresholds,
            plot_rank=plot_rank,
        )
        results.append(result)
        if abs(float(fraction) - table_holdout) < 1e-9:
            table_result = result

    if table_result is None:
        table_result = results[len(results) // 2] if results else None
    if table_result is None:
        raise RuntimeError("No holdout splits were evaluated.")

    rows = build_table_rows(table_result)
    _log_table(rows, table_result["holdout_fraction"])

    out_dir = resolve_path(h2h_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    table_df = pd.DataFrame(rows)
    table_csv = out_dir / "table.csv"
    table_md = out_dir / "table.md"
    table_df.to_csv(table_csv, index=False)
    table_md.write_text(format_markdown_table(rows), encoding="utf-8")

    plot_path = out_dir / "fcw_vs_threshold.png"
    plot_fcw_vs_threshold(table_result, plot_path, rank=plot_rank, subset=plot_subset)

    payload = {
        "table_holdout_fraction": table_result["holdout_fraction"],
        "default_confidence_threshold": threshold,
        "plot_rank": plot_rank,
        "plot_subset": plot_subset,
        "table": rows,
        "splits": results,
    }
    json_path = out_dir / "head_to_head.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2)

    logger.info("Wrote %s", table_csv)
    logger.info("Wrote %s", table_md)
    logger.info("Wrote %s", plot_path)
    logger.info("Wrote %s", json_path)
    return payload


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
