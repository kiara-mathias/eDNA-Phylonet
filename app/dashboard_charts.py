"""Pure helpers for the Step 4 dashboard panels (calibration + FCW).

Kept out of ``dashboard.py`` so the reliability / false-confident-wrong
math can be unit-tested without Streamlit. Charts consume the JSON written
by ``src.eval.validate_calibration`` and ``src.eval.head_to_head`` -- they
do not re-fit models.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.eval.calibration import RankCalibrationReport, evaluate_rank_calibration

_RANKS = ("species", "genus", "family", "order")
_METHOD_LABELS = {
    "ours": "ours",
    "blast": "BLAST",
    "naive_bayes": "Naive Bayes",
    "nearest_neighbor": "1-NN",
}


def reliability_frame(reports: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Bins from calibration.json ``reports`` → a plottable table."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        rank = report["rank"]
        for bin_row in report.get("bins", []):
            if not bin_row.get("count"):
                continue
            mean_conf = bin_row.get("mean_confidence")
            acc = bin_row.get("accuracy")
            if mean_conf is None or acc is None:
                continue
            if not np.isfinite(mean_conf) or not np.isfinite(acc):
                continue
            rows.append(
                {
                    "rank": rank,
                    "mean_confidence": float(mean_conf),
                    "accuracy": float(acc),
                    "count": int(bin_row["count"]),
                    "ece": float(report.get("ece", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def live_reliability_reports(
    predictions_df: pd.DataFrame,
    n_bins: int,
    ranks: tuple[str, ...] = _RANKS,
) -> list[RankCalibrationReport]:
    """Re-bin held-out (confidence, correct) pairs -- the live calibration panel."""
    reports = []
    for rank in ranks:
        conf_col = f"confidence_{rank}"
        correct_col = f"correct_{rank}"
        if conf_col not in predictions_df.columns or correct_col not in predictions_df.columns:
            continue
        reports.append(
            evaluate_rank_calibration(
                predictions_df[conf_col].to_numpy(dtype=np.float64),
                predictions_df[correct_col].to_numpy(dtype=np.float64),
                rank,
                n_bins=n_bins,
            )
        )
    return reports


def reports_to_dicts(reports: Sequence[RankCalibrationReport]) -> list[dict[str, Any]]:
    return [
        {
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
        for report in reports
    ]


def fcw_curve_frame(
    payload: dict[str, Any],
    *,
    rank: str,
    subset: str,
    methods: Sequence[str],
) -> pd.DataFrame:
    """Long-form FCW vs. threshold for the selected methods."""
    holdout = payload.get("table_holdout_fraction")
    splits = payload.get("splits") or []
    result = None
    for split in splits:
        if holdout is not None and abs(float(split.get("holdout_fraction", -1)) - float(holdout)) < 1e-9:
            result = split
            break
    if result is None and splits:
        result = splits[0]
    if result is None:
        return pd.DataFrame(columns=["method", "label", "threshold", "fcw"])

    thresholds = [float(t) for t in result.get("thresholds", [])]
    rows: list[dict[str, Any]] = []
    systems = result.get("systems") or {}
    for method in methods:
        scores = systems.get(method)
        if not scores:
            continue
        curve = scores.get(subset, {}).get(rank, {}).get("fcw_curve")
        if not curve:
            continue
        label = _METHOD_LABELS.get(method, method)
        for threshold, value in zip(thresholds, curve):
            rows.append(
                {
                    "method": method,
                    "label": label,
                    "threshold": float(threshold),
                    "fcw": float(value) if value is not None and np.isfinite(value) else None,
                }
            )
    return pd.DataFrame(rows)


def fcw_at_threshold(curve_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Linear interpolation of each method's FCW curve at ``threshold``."""
    if curve_df.empty:
        return pd.DataFrame(columns=["method", "label", "threshold", "fcw"])
    rows = []
    for method, group in curve_df.groupby("method", sort=False):
        xs = group["threshold"].to_numpy(dtype=np.float64)
        ys = group["fcw"].to_numpy(dtype=np.float64)
        order = np.argsort(xs)
        value = float(np.interp(threshold, xs[order], ys[order]))
        rows.append(
            {
                "method": method,
                "label": group["label"].iloc[0],
                "threshold": float(threshold),
                "fcw": value,
            }
        )
    return pd.DataFrame(rows)


def format_sequence_html(sequence: str, max_len: int = 180) -> str:
    """Monospace, base-colored DNA for the dashboard (and the BLAST gallery)."""
    colors = {"A": "#1b7a3d", "C": "#2a6f97", "G": "#c98a12", "T": "#b3261e", "N": "#5f6368"}
    shown = sequence[:max_len]
    parts: list[str] = []
    for i, base in enumerate(shown):
        color = colors.get(base.upper(), "#3c4043")
        parts.append(f'<span style="color:{color}">{base}</span>')
        if (i + 1) % 60 == 0 and i + 1 < len(shown):
            parts.append("<br/>")
    extra = f'<span style="color:#5f6368"> … +{len(sequence) - max_len} bp</span>' if len(sequence) > max_len else ""
    return (
        '<div class="edna-seq" style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;'
        'font-size:0.82rem;letter-spacing:0.06em;line-height:1.7;word-break:break-all;">'
        f"{''.join(parts)}{extra}</div>"
    )


def confidence_gradient_css(confidence: float) -> str:
    """Background interpolating red → amber → green with predicted confidence."""
    conf = float(np.clip(confidence, 0.0, 1.0))
    if conf < 0.5:
        t = conf / 0.5
        r = int(179 + (201 - 179) * t)
        g = int(38 + (138 - 38) * t)
        b = int(30 + (18 - 30) * t)
    else:
        t = (conf - 0.5) / 0.5
        r = int(201 + (27 - 201) * t)
        g = int(138 + (122 - 138) * t)
        b = int(18 + (61 - 18) * t)
    return f"rgb({r},{g},{b})"
