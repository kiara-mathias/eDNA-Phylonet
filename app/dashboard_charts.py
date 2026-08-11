"""Pure helpers for dashboard panels (calibration, FCW, holdout bars).

Kept out of ``dashboard.py`` so the math can be unit-tested without
Streamlit. Charts consume JSON written by ``src.eval.validate_calibration``,
``src.eval.head_to_head``, and ``src.eval.benchmark`` -- they do not re-fit
models.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.eval.calibration import RankCalibrationReport, evaluate_rank_calibration
from theme.tokens import CORAL, INK, NAVY, RANK_BAND, SAND, SEAFOAM, TEAL

_RANKS = ("species", "genus", "family", "order")
_METHOD_LABELS = {
    "ours": "ours",
    "blast": "BLAST",
    "naive_bayes": "Naive Bayes",
    "nearest_neighbor": "1-NN",
}
_METHOD_COLOR = {
    "ours": SEAFOAM,
    "blast": CORAL,
    "naive_bayes": TEAL,
    "nearest_neighbor": NAVY,
}
_METRIC_ORDER = tuple(f"{rank} {kind}" for rank in _RANKS for kind in ("coverage", "accuracy"))


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


def holdout_percent_options(results: Sequence[dict[str, Any]]) -> list[int]:
    return sorted({int(round(float(split["holdout_fraction"]) * 100)) for split in results})


def split_at_holdout(results: Sequence[dict[str, Any]], percent: int) -> dict[str, Any] | None:
    for split in results:
        if int(round(float(split["holdout_fraction"]) * 100)) == int(percent):
            return split
    return None


def benchmark_bar_frame(split: dict[str, Any]) -> pd.DataFrame:
    """Long-form coverage / accuracy-among-answered for one holdout split."""
    rows: list[dict[str, Any]] = []
    systems = split.get("systems") or {}
    for method, scores in systems.items():
        if not scores:
            continue
        label = _METHOD_LABELS.get(method, method)
        for rank in _RANKS:
            rank_scores = scores.get(rank) or {}
            coverage = rank_scores.get("coverage")
            accuracy = rank_scores.get("accuracy_among_answered")
            if coverage is not None and np.isfinite(coverage):
                rows.append(
                    {
                        "method": method,
                        "label": label,
                        "metric": f"{rank} coverage",
                        "value": float(coverage),
                    }
                )
            if accuracy is not None and np.isfinite(accuracy):
                rows.append(
                    {
                        "method": method,
                        "label": label,
                        "metric": f"{rank} accuracy",
                        "value": float(accuracy),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["metric"] = pd.Categorical(frame["metric"], categories=list(_METRIC_ORDER), ordered=True)
    return frame.sort_values(["metric", "method"])


def plotly_theme_layout(**overrides: Any) -> dict[str, Any]:
    layout = dict(
        paper_bgcolor=SAND,
        plot_bgcolor=SAND,
        font=dict(family='Inter, "Segoe UI", sans-serif', color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor=SAND, borderwidth=0),
        margin=dict(l=48, r=16, t=48, b=72),
        hoverlabel=dict(bgcolor=SAND, font_size=12, font_family='Inter, "Segoe UI", sans-serif'),
    )
    layout.update(overrides)
    return layout


def build_reliability_figure(frame: pd.DataFrame) -> go.Figure:
    """Reliability diagram: perfect diagonal + one line per rank (legend-toggle)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect",
            line=dict(color=NAVY, width=1.2, dash="dash"),
            opacity=0.4,
            hoverinfo="skip",
        )
    )
    if frame.empty:
        fig.update_layout(**plotly_theme_layout(xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])))
        return fig
    for rank in _RANKS:
        group = frame.loc[frame["rank"] == rank]
        if group.empty:
            continue
        ece = group["ece"].iloc[0] if "ece" in group.columns else float("nan")
        label = str(rank) + (f"  ECE {ece:.3f}" if np.isfinite(ece) else "")
        ordered = group.sort_values("mean_confidence")
        color = RANK_BAND.get(str(rank), TEAL)
        fig.add_trace(
            go.Scatter(
                x=list(ordered["mean_confidence"]),
                y=list(ordered["accuracy"]),
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.2),
                marker=dict(size=7, color=color),
                hovertemplate=(
                    f"{rank}<br>confidence=%{{x:.2f}}<br>accuracy=%{{y:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        **plotly_theme_layout(
            height=440,
            xaxis=dict(
                range=[0, 1],
                title="Predicted confidence (bin mean)",
                gridcolor="rgba(10,31,46,0.12)",
                zeroline=False,
                tickformat=".0%",
            ),
            yaxis=dict(
                range=[0, 1],
                title="Observed accuracy",
                gridcolor="rgba(10,31,46,0.12)",
                zeroline=False,
                tickformat=".0%",
            ),
        )
    )
    return fig


def build_benchmark_bar_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    for method, group in frame.groupby("method", sort=False):
        ordered = group.sort_values("metric")
        fig.add_trace(
            go.Bar(
                x=list(ordered["metric"]),
                y=list(ordered["value"]),
                name=str(ordered["label"].iloc[0]),
                marker_color=_METHOD_COLOR.get(str(method), TEAL),
                hovertemplate="%{x}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        **plotly_theme_layout(
            barmode="group",
            bargap=0.28,
            bargroupgap=0.1,
            yaxis=dict(
                range=[0, 1],
                title=None,
                gridcolor="rgba(10,31,46,0.12)",
                zeroline=False,
                tickformat=".0%",
            ),
            xaxis=dict(title=None, tickangle=-28, automargin=True),
        )
    )
    return fig


def fcw_callout_row(at_t: pd.DataFrame) -> dict[str, Any] | None:
    """Prefer ``ours`` as the headline false-confident-wrong rate."""
    if at_t.empty:
        return None
    ours = at_t.loc[at_t["method"] == "ours"]
    row = ours.iloc[0] if not ours.empty else at_t.iloc[0]
    return {"method": row["method"], "label": row["label"], "fcw": float(row["fcw"])}


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
