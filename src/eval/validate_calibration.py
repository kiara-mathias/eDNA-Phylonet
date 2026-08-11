"""CLI: prove per-rank confidence is calibrated on clade-exclusion holdouts.

Usage::

    python -m src.eval.validate_calibration --config configs/eval.yaml

For each 30/50/70% genus holdout split, fits the encoder +
``HierarchicalFallback`` on train, calibrates distance thresholds on val,
then for every held-out (test) query records per-rank confidence and
whether the nearest label at that rank matches the truth.

Aggregates those (confidence, correct) pairs across holdouts, bins by
confidence to build a reliability diagram (one curve per rank), and
reports Expected Calibration Error (ECE) per rank.

Decision: if any rank's ECE exceeds ``calibration.ece_threshold`` (default
0.1), fit Platt / isotonic maps via leave-one-holdout-out on the other
holdouts' test predictions (val-fit does not transfer to zero-shot
queries), apply them to the held-out confidences, and re-report ECE + a
second reliability diagram. Only ranks that failed the threshold are
remapped; cascade commit decisions stay distance-threshold based.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, resolve_path  # noqa: E402
from src.eval.calibration import (  # noqa: E402
    evaluate_rank_calibration,
    fit_rank_calibrators,
    leave_one_holdout_recalibrate,
    plot_reliability_diagram,
)
from src.fallback.novelty import HierarchicalFallback  # noqa: E402
from src.features.encoder import build_encoder, encoder_name  # noqa: E402

logger = logging.getLogger(__name__)

_RANKS = ("species", "genus", "family", "order")


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def _empty_rank_arrays() -> dict[str, list[float]]:
    return {rank: [] for rank in _RANKS}


def _append_prediction_rows(
    *,
    predictions: list[Any],
    labels_df: pd.DataFrame,
    holdout_fraction: float,
    split_name: str,
    process_ids: list[str],
    confidence_by_rank: dict[str, list[float]],
    correct_by_rank: dict[str, list[float]],
    rows: list[dict[str, Any]],
) -> None:
    for i, pred in enumerate(predictions):
        row: dict[str, Any] = {
            "process_id": process_ids[i],
            "holdout_fraction": holdout_fraction,
            "split": split_name,
            "predicted_rank": pred.predicted_rank,
            "is_novel": bool(pred.is_novel),
        }
        for rank in _RANKS:
            nearest = pred.nearest[rank]
            conf = float(pred.confidence[rank])
            correct = float(nearest == labels_df.iloc[i][rank])
            confidence_by_rank[rank].append(conf)
            correct_by_rank[rank].append(correct)
            row[f"nearest_{rank}"] = nearest
            row[f"true_{rank}"] = labels_df.iloc[i][rank]
            row[f"confidence_{rank}"] = conf
            row[f"correct_{rank}"] = int(correct)
        rows.append(row)


def _to_numpy_dict(lists: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {rank: np.asarray(vals, dtype=np.float64) for rank, vals in lists.items()}


def _reports_from_arrays(
    confidence_by_rank: dict[str, np.ndarray],
    correct_by_rank: dict[str, np.ndarray],
    n_bins: int,
) -> list[Any]:
    return [
        evaluate_rank_calibration(confidence_by_rank[rank], correct_by_rank[rank], rank, n_bins=n_bins)
        for rank in _RANKS
    ]


def _report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "rank": report.rank,
        "n": report.n,
        "ece": report.ece,
        "bins": [
            {
                "bin_index": b.bin_index,
                "conf_low": b.conf_low,
                "conf_high": b.conf_high,
                "mean_confidence": b.mean_confidence,
                "accuracy": b.accuracy,
                "count": b.count,
            }
            for b in report.bins
        ],
    }


def collect_split_predictions(
    split: dict[str, Any],
    sequences_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[dict[str, Any]]]:
    """Fit one holdout model; return val/test confidence+correctness lists + row dump."""
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

    val_conf: dict[str, list[float]] = _empty_rank_arrays()
    val_correct: dict[str, list[float]] = _empty_rank_arrays()
    test_conf: dict[str, list[float]] = _empty_rank_arrays()
    test_correct: dict[str, list[float]] = _empty_rank_arrays()
    rows: list[dict[str, Any]] = []

    _append_prediction_rows(
        predictions=val_predictions,
        labels_df=val_df,
        holdout_fraction=split["holdout_fraction"],
        split_name="val",
        process_ids=list(val_df["process_id"]),
        confidence_by_rank=val_conf,
        correct_by_rank=val_correct,
        rows=rows,
    )
    _append_prediction_rows(
        predictions=test_predictions,
        labels_df=test_df,
        holdout_fraction=split["holdout_fraction"],
        split_name="test",
        process_ids=list(test_df["process_id"]),
        confidence_by_rank=test_conf,
        correct_by_rank=test_correct,
        rows=rows,
    )
    return val_conf, val_correct, test_conf, test_correct, rows


def _merge_lists(
    dest: dict[str, list[float]],
    src: dict[str, list[float]],
) -> None:
    for rank in _RANKS:
        dest[rank].extend(src[rank])


def run(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    splitter_cfg = config["splitter"]
    calib_cfg = config.get("calibration", {})
    n_bins = int(calib_cfg.get("n_bins", 10))
    ece_threshold = float(calib_cfg.get("ece_threshold", 0.1))
    method = str(calib_cfg.get("method", "isotonic"))
    auto_recalibrate = bool(calib_cfg.get("auto_recalibrate", True))

    sequences_df = pd.read_parquet(resolve_path(data_cfg["sequences_path"]))
    splits_dir = resolve_path(splitter_cfg["splits_dir"])
    out_dir = resolve_path(calib_cfg.get("output_dir", "data/eval_results/calibration"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_val_conf = _empty_rank_arrays()
    pooled_val_correct = _empty_rank_arrays()
    pooled_test_conf = _empty_rank_arrays()
    pooled_test_correct = _empty_rank_arrays()
    pooled_test_holdout: list[float] = []
    all_rows: list[dict[str, Any]] = []
    per_holdout: list[dict[str, Any]] = []

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
            "Collecting confidence+correctness for holdout_%d (%d/%d genera held out)...",
            pct,
            split["n_genera_held_out"],
            split["n_genera_total"],
        )
        val_conf, val_correct, test_conf, test_correct, rows = collect_split_predictions(
            split, sequences_df, config
        )
        _merge_lists(pooled_val_conf, val_conf)
        _merge_lists(pooled_val_correct, val_correct)
        _merge_lists(pooled_test_conf, test_conf)
        _merge_lists(pooled_test_correct, test_correct)
        pooled_test_holdout.extend([float(split["holdout_fraction"])] * len(test_conf["species"]))
        all_rows.extend(rows)

        holdout_test_conf = _to_numpy_dict(test_conf)
        holdout_test_correct = _to_numpy_dict(test_correct)
        holdout_reports = _reports_from_arrays(holdout_test_conf, holdout_test_correct, n_bins)
        per_holdout.append(
            {
                "holdout_fraction": split["holdout_fraction"],
                "n_test": len(test_conf["species"]),
                "n_val": len(val_conf["species"]),
                "ece_by_rank": {r.rank: r.ece for r in holdout_reports},
            }
        )
        logger.info(
            "  holdout_%d test ECE: %s",
            pct,
            ", ".join(f"{r.rank}={r.ece:.4f}" for r in holdout_reports),
        )

    test_conf_np = _to_numpy_dict(pooled_test_conf)
    test_correct_np = _to_numpy_dict(pooled_test_correct)
    val_conf_np = _to_numpy_dict(pooled_val_conf)
    val_correct_np = _to_numpy_dict(pooled_val_correct)
    holdout_groups = np.asarray(pooled_test_holdout, dtype=np.float64)

    raw_reports = _reports_from_arrays(test_conf_np, test_correct_np, n_bins)
    raw_ece = {r.rank: r.ece for r in raw_reports}
    bad_ranks = [rank for rank, ece in raw_ece.items() if ece > ece_threshold]

    logger.info(
        "Pooled held-out ECE (raw): %s",
        ", ".join(f"{rank}={ece:.4f}" for rank, ece in raw_ece.items()),
    )

    predictions_path = out_dir / "heldout_predictions.parquet"
    pd.DataFrame(all_rows).to_parquet(predictions_path, index=False)
    logger.info("Wrote per-query confidence+correctness -> %s", predictions_path)

    raw_plot_path = out_dir / "reliability_raw.png"
    plot_reliability_diagram(
        raw_reports,
        raw_plot_path,
        title="Reliability diagram (raw confidence, held-out genera)",
    )
    logger.info("Wrote %s", raw_plot_path)

    recalibrated_reports = None
    recalibrated_ece: dict[str, float] | None = None
    calibrator_method_used: str | None = None
    decision = "pass"
    calibrators_path: str | None = None

    if bad_ranks and auto_recalibrate:
        decision = "recalibrate"
        calibrator_method_used = method
        logger.info(
            "ECE > %.2f for ranks %s -- leave-one-holdout-out %s recalibration "
            "(val-fit does not transfer to clade-exclusion queries).",
            ece_threshold,
            bad_ranks,
            method,
        )
        # Keep well-calibrated ranks (species/genus) unchanged; only remap
        # ranks that failed the ECE gate.
        calibrated_test_conf = {rank: test_conf_np[rank].copy() for rank in _RANKS}
        for rank in bad_ranks:
            calibrated_test_conf[rank] = leave_one_holdout_recalibrate(
                test_conf_np[rank],
                test_correct_np[rank],
                holdout_groups,
                method=method,  # type: ignore[arg-type]
            )

        recalibrated_reports = _reports_from_arrays(calibrated_test_conf, test_correct_np, n_bins)
        recalibrated_ece = {r.rank: r.ece for r in recalibrated_reports}
        still_bad = [rank for rank, ece in recalibrated_ece.items() if ece > ece_threshold]
        if still_bad:
            decision = "fail_after_recalibrate"
            logger.warning(
                "After %s recalibration, ECE still > %.2f for: %s",
                method,
                ece_threshold,
                still_bad,
            )
        else:
            decision = "pass_after_recalibrate"
            logger.info("All ranks ECE <= %.2f after %s recalibration.", ece_threshold, method)

        calib_plot_path = out_dir / "reliability_recalibrated.png"
        plot_reliability_diagram(
            recalibrated_reports,
            calib_plot_path,
            title=f"Reliability diagram (LOHO {method}-calibrated, held-out genera)",
        )
        logger.info("Wrote %s", calib_plot_path)

        # Deployment calibrators: fit on all held-out test rows for the failing
        # ranks (zero-shot regime), and on val for every rank (in-distribution).
        import pickle

        heldout_calibrators = fit_rank_calibrators(
            {rank: test_conf_np[rank] for rank in bad_ranks},
            {rank: test_correct_np[rank] for rank in bad_ranks},
            method=method,  # type: ignore[arg-type]
            ranks=tuple(bad_ranks),
        )
        val_calibrators = fit_rank_calibrators(val_conf_np, val_correct_np, method=method)  # type: ignore[arg-type]
        calibrators_path = str(out_dir / f"calibrators_{method}.pkl")
        with open(calibrators_path, "wb") as fh:
            pickle.dump(
                {
                    "method": method,
                    "bad_ranks": bad_ranks,
                    "heldout_fit": heldout_calibrators,
                    "val_fit": val_calibrators,
                },
                fh,
            )
        logger.info("Wrote %s", calibrators_path)
    elif bad_ranks:
        decision = "fail"
        logger.warning(
            "ECE > %.2f for ranks %s and auto_recalibrate=false -- not fixing.",
            ece_threshold,
            bad_ranks,
        )
    else:
        logger.info("All ranks ECE <= %.2f -- raw confidence is acceptably calibrated.", ece_threshold)

    summary = {
        "encoder": encoder_name(config.get("encoder", {})),
        "n_bins": n_bins,
        "ece_threshold": ece_threshold,
        "method": method,
        "auto_recalibrate": auto_recalibrate,
        "decision": decision,
        "bad_ranks_raw": bad_ranks,
        "n_heldout_predictions": int(len(test_conf_np["species"])),
        "n_val_predictions": int(len(val_conf_np["species"])),
        "per_holdout": per_holdout,
        "raw": {
            "ece_by_rank": raw_ece,
            "reports": [_report_to_dict(r) for r in raw_reports],
            "reliability_plot": str(raw_plot_path),
        },
        "recalibrated": None
        if recalibrated_reports is None
        else {
            "method": calibrator_method_used,
            "protocol": "leave_one_holdout",
            "ece_by_rank": recalibrated_ece,
            "reports": [_report_to_dict(r) for r in recalibrated_reports],
            "reliability_plot": str(out_dir / "reliability_recalibrated.png"),
            "calibrators_path": calibrators_path,
        },
        "predictions_path": str(predictions_path),
    }

    summary_path = out_dir / "calibration.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Wrote %s", summary_path)
    logger.info("Decision: %s", decision)
    return summary


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
