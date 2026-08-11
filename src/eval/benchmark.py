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
- **coverage_ci** / **accuracy_among_answered_ci**: bootstrap percentile
  intervals over resampled test queries (configurable via
  ``benchmark.n_bootstrap`` / ``benchmark.ci_level``).

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
from src.features.encoder import build_encoder  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")
_Prediction = BaselinePrediction | FallbackPrediction


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    answered = np.array([v is not None for v in y_pred], dtype=bool)
    coverage = float(answered.mean()) if len(y_pred) else float("nan")
    if answered.any():
        accuracy = float((y_pred[answered] == y_true[answered]).mean())
    else:
        accuracy = float("nan")
    return coverage, accuracy


def _score(
    predictions: Sequence[_Prediction],
    test_df: pd.DataFrame,
    rank: str,
    *,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Point estimates plus bootstrap percentile CIs for one rank."""
    y_true = test_df[rank].to_numpy()
    y_pred = np.array([getattr(p, rank) for p in predictions], dtype=object)
    coverage, accuracy = _point_metrics(y_true, y_pred)

    result: dict[str, Any] = {
        "coverage": _finite_or_none(coverage),
        "accuracy_among_answered": _finite_or_none(accuracy),
        "coverage_ci": None,
        "accuracy_among_answered_ci": None,
        "n_bootstrap": int(n_bootstrap),
        "ci_level": float(ci_level),
    }

    n = len(y_pred)
    if n_bootstrap <= 0 or n == 0 or rng is None:
        return result

    alpha = (1.0 - ci_level) / 2.0
    cov_samples = np.empty(n_bootstrap, dtype=np.float64)
    acc_samples = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cov_samples[i], acc_samples[i] = _point_metrics(y_true[idx], y_pred[idx])

    result["coverage_ci"] = [
        _finite_or_none(float(np.percentile(cov_samples, 100.0 * alpha))),
        _finite_or_none(float(np.percentile(cov_samples, 100.0 * (1.0 - alpha)))),
    ]
    finite_acc = acc_samples[np.isfinite(acc_samples)]
    if len(finite_acc) == 0:
        result["accuracy_among_answered_ci"] = [None, None]
    else:
        result["accuracy_among_answered_ci"] = [
            _finite_or_none(float(np.percentile(finite_acc, 100.0 * alpha))),
            _finite_or_none(float(np.percentile(finite_acc, 100.0 * (1.0 - alpha)))),
        ]
    return result


def _score_system(
    predictions: Sequence[_Prediction],
    test_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    return {
        rank: _score(
            predictions,
            test_df,
            rank,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            rng=rng,
        )
        for rank in _RANKS
    }


def evaluate_split(split: dict[str, Any], sequences_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    encoder_cfg = config["encoder"]
    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]
    baselines_cfg = config["baselines"]
    benchmark_cfg = config.get("benchmark", {})
    n_bootstrap = int(benchmark_cfg.get("n_bootstrap", 1000))
    ci_level = float(benchmark_cfg.get("ci_level", 0.95))
    bootstrap_seed = int(benchmark_cfg.get("seed", 42))

    by_id = sequences_df.set_index("process_id")
    train_df = by_id.loc[split["train_process_ids"]].reset_index()
    val_df = by_id.loc[split["val_process_ids"]].reset_index()
    test_df = by_id.loc[split["test_process_ids"]].reset_index()

    encoder = build_encoder(encoder_cfg)
    train_embeddings = encoder.fit_transform(train_df["sequence"], train_df["species"])
    val_embeddings = encoder.transform(val_df["sequence"])
    test_embeddings = encoder.transform(test_df["sequence"])

    ours = HierarchicalFallback(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
        rank_percentile=fallback_cfg["rank_percentile"],
        sample_count_prior=float(fallback_cfg.get("sample_count_prior", 5.0)),
        n_nearest_relatives=int(fallback_cfg.get("n_nearest_relatives", 5)),
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

    # Independent RNG streams per system so adding/removing a baseline
    # doesn't reshuffle another system's bootstrap draws.
    score_kwargs = {"n_bootstrap": n_bootstrap, "ci_level": ci_level}
    systems: dict[str, dict[str, dict[str, Any]] | None] = {
        "ours": _score_system(
            ours_predictions, test_df, rng=np.random.default_rng(bootstrap_seed + 1), **score_kwargs
        ),
        "naive_bayes": _score_system(
            naive_bayes_predictions, test_df, rng=np.random.default_rng(bootstrap_seed + 2), **score_kwargs
        ),
        "nearest_neighbor": _score_system(
            nearest_neighbor_predictions, test_df, rng=np.random.default_rng(bootstrap_seed + 3), **score_kwargs
        ),
    }

    if BlastBaseline.is_available():
        blast_cfg = baselines_cfg["blast"]
        pct = round(split["holdout_fraction"] * 100)
        db_dir = resolve_path(blast_cfg["db_dir"]) / f"holdout_{pct}"
        blast = BlastBaseline(db_dir=db_dir, evalue=blast_cfg["evalue"], min_pident=blast_cfg["min_pident"])
        blast.fit(train_df)
        blast_predictions = blast.predict(test_df)
        systems["blast"] = _score_system(
            blast_predictions, test_df, rng=np.random.default_rng(bootstrap_seed + 4), **score_kwargs
        )
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


def _fmt_metric(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "nan"


def _fmt_ci(ci: list[float | None] | None) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return ""
    return f"[{ci[0]:.2f}-{ci[1]:.2f}]"


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
            f"{rank}=cov:{_fmt_metric(scores[rank]['coverage'])}"
            f"{_fmt_ci(scores[rank].get('coverage_ci'))}"
            f"/acc:{_fmt_metric(scores[rank]['accuracy_among_answered'])}"
            f"{_fmt_ci(scores[rank].get('accuracy_among_answered_ci'))}"
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
        json.dump(results, fh, indent=2, allow_nan=False)
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
