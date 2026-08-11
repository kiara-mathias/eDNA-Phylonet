"""Streamlit dashboard: a specimen record, not a BI layout.

The page is a single vertical scroll. Color is taxonomic confidence as
ocean depth -- species (trustworthy) is shallow seafoam, order (the last
fallback) is deep navy -- the same cascade ``HierarchicalFallback`` uses.
Only the depth-tree reveal animates; everything else stays quiet.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `streamlit run app/dashboard.py`

from app.pipeline import (  # noqa: E402
    Pipeline,
    RelativeHit,
    SequenceParseError,
    classify_sequence,
    clean_sequence_text,
    load_app_config,
    load_pipeline,
    nearest_relatives,
)
from src.common import resolve_path  # noqa: E402
from src.eval.compare_encoders import false_confident_wrong_rate  # noqa: E402
from src.fallback.novelty import FallbackPrediction  # noqa: E402

_RANKS = ("species", "genus", "family", "order")
_RANK_INDEX = {rank: i for i, rank in enumerate(_RANKS)}

# Ocean-depth tokens. Deeper = less certain, matching the fallback cascade.
_SAND = "#F7F4EC"
_NAVY = "#0A1F2E"
_TEAL = "#1B5E6C"
_SEAFOAM = "#4FA8A0"
_CORAL = "#E8734A"
_INK = "#2C2C2C"
_RANK_BAND = {
    "species": "#4FA8A0",
    "genus": "#2A7A82",
    "family": "#1B5E6C",
    "order": "#0A1F2E",
}
_RANK_INK = {
    "species": "#0A1F2E",
    "genus": "#F7F4EC",
    "family": "#F7F4EC",
    "order": "#F7F4EC",
}
_TREE_REVEAL_MS = 250

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {{
  background: {_SAND};
  color: {_INK};
  font-family: Inter, "Segoe UI", sans-serif;
}}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer {{
  visibility: hidden;
  height: 0;
}}
[data-testid="stSidebar"] {{ display: none; }}
.block-container {{
  max-width: 880px;
  padding-top: 1.25rem;
  padding-bottom: 4rem;
}}
h1, h2, h3, .record-kicker, .panel-kicker {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: 0.04em;
}}
.record-header {{
  background: {_NAVY};
  color: {_SAND};
  padding: 1.35rem 1.4rem 1.2rem;
  margin: 0 0 1.5rem 0;
}}
.record-kicker {{
  font-size: 0.68rem;
  text-transform: uppercase;
  opacity: 0.7;
  margin: 0 0 0.35rem 0;
}}
.record-header h1 {{
  font-size: 1.35rem;
  font-weight: 500;
  margin: 0 0 0.45rem 0;
  color: {_SAND};
}}
.record-header p {{
  margin: 0;
  font-family: Inter, sans-serif;
  font-size: 0.92rem;
  line-height: 1.45;
  color: {_SAND};
  opacity: 0.88;
}}
.record-meta {{
  margin-top: 0.75rem;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  opacity: 0.65;
}}
textarea {{
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  letter-spacing: 0.12em !important;
  font-size: 0.82rem !important;
  background: {_SAND} !important;
  color: {_INK} !important;
}}
div[data-testid="stTextArea"] label, div[data-testid="stTextInput"] label {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.stButton > button {{
  background: {_SEAFOAM};
  color: {_NAVY};
  border: 0;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: 0.06em;
  font-weight: 500;
}}
.stButton > button:hover {{
  background: {_TEAL};
  color: {_SAND};
}}
div[data-testid="stAlert"] {{
  background: transparent;
  border: 1px solid {_TEAL};
  color: {_INK};
}}
.specimen-seq {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: 0.16em;
  font-size: 0.78rem;
  line-height: 1.7;
  color: {_NAVY};
  border: 1px solid {_NAVY}22;
  padding: 0.7rem 0.85rem;
  margin: 0.4rem 0 1.2rem 0;
  word-break: break-all;
}}
.depth-well {{
  margin: 0.25rem 0 1.75rem 0;
}}
.panel-kicker {{
  font-size: 0.68rem;
  text-transform: uppercase;
  color: {_NAVY};
  margin: 0 0 0.35rem 0;
}}
.panel-lede {{
  font-size: 0.84rem;
  color: {_INK};
  opacity: 0.78;
  margin: 0 0 0.85rem 0;
}}
.depth-ranks {{
  list-style: none;
  margin: 0;
  padding: 0;
}}
.depth-rank {{
  display: grid;
  grid-template-columns: 6.5rem 1fr auto;
  gap: 0.75rem;
  align-items: baseline;
  padding: 0.85rem 1rem;
  animation: depth-reveal {_TREE_REVEAL_MS}ms ease-out both;
  animation-delay: var(--delay);
}}
.depth-rank .rank-name {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  opacity: 0.8;
}}
.depth-rank .taxon {{
  font-family: Inter, sans-serif;
  font-size: 1.02rem;
  font-weight: 500;
}}
.depth-rank .conf {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
}}
.depth-rank.is-skipped, .depth-rank.is-abyss {{
  animation-name: depth-reveal-dim;
}}
.depth-rank.is-committed {{
  border-left: 3px solid {_SAND};
}}
.abyss-note {{
  background: {_NAVY};
  color: {_SAND};
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 0.65rem 1rem;
  opacity: 0.85;
  animation: depth-reveal {_TREE_REVEAL_MS}ms ease-out both;
  animation-delay: calc(4 * {_TREE_REVEAL_MS}ms);
}}
@keyframes depth-reveal {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to {{ opacity: 1; transform: none; }}
}}
@keyframes depth-reveal-dim {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to {{ opacity: 0.28; transform: none; }}
}}
.relative-row, .lie-card {{
  border-top: 1px solid {_NAVY}18;
  padding: 0.55rem 0;
  font-size: 0.9rem;
}}
.relative-row .name, .lie-card .name {{
  font-family: Inter, sans-serif;
  font-weight: 500;
}}
.mono-quiet {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  opacity: 0.7;
}}
.coral-flag {{
  color: {_CORAL};
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}}
.lie-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}}
@media (max-width: 700px) {{
  .lie-grid {{ grid-template-columns: 1fr; }}
  .depth-rank {{ grid-template-columns: 1fr; gap: 0.2rem; }}
}}
"""


@st.cache_resource(show_spinner="Fitting encoder + classifier + fallback on the known reference data...")
def _cached_pipeline(config_path: str) -> Pipeline | None:
    config = load_app_config(config_path)
    try:
        return load_pipeline(config)
    except FileNotFoundError:
        return None


def _inject_theme() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def rank_visual_state(prediction: FallbackPrediction, rank: str) -> str:
    """``skipped`` (shallower than the commit), ``committed``, or ``abyss`` (deeper)."""
    predicted = prediction.predicted_rank
    if predicted is None:
        return "abyss"
    if _RANK_INDEX[rank] < _RANK_INDEX[predicted]:
        return "skipped"
    if rank == predicted:
        return "committed"
    return "abyss"


def _taxon_at_rank(prediction: FallbackPrediction, rank: str) -> str:
    committed = getattr(prediction, rank)
    if committed:
        return str(committed)
    nearest = prediction.nearest.get(rank)
    return str(nearest) if nearest else "—"


def build_depth_tree_html(prediction: FallbackPrediction) -> str:
    """Vertical water column: species at the surface, order at the bottom."""
    rows: list[str] = []
    for i, rank in enumerate(_RANKS):
        state = rank_visual_state(prediction, rank)
        confidence = prediction.confidence.get(rank, 0.0)
        taxon = html.escape(_taxon_at_rank(prediction, rank))
        bg = _RANK_BAND[rank]
        fg = _RANK_INK[rank]
        delay = f"{i * _TREE_REVEAL_MS}ms"
        rows.append(
            f'<li class="depth-rank is-{html.escape(state)}" '
            f'style="background:{bg};color:{fg};--delay:{delay}">'
            f'<span class="rank-name">{html.escape(rank)}</span>'
            f'<span class="taxon">{taxon}</span>'
            f'<span class="conf">{confidence * 100:.0f}%</span>'
            f"</li>"
        )

    stop = prediction.predicted_rank or "unresolved"
    abyss = (
        "unresolved water — no trustworthy rank"
        if prediction.predicted_rank is None
        else f"below {stop} · not trustworthy"
    )
    return (
        '<section class="depth-well">'
        '<p class="panel-kicker">Taxonomic depth</p>'
        '<p class="panel-lede">Shallow water is a species-level call. '
        "Each band darker is one rank of fallback. The cascade stops at the "
        "last trustworthy rank; below that is unresolved water.</p>"
        f'<ol class="depth-ranks">{"".join(rows)}</ol>'
        f'<div class="abyss-note">{html.escape(abyss)}</div>'
        "</section>"
    )


def format_specimen_sequence(sequence: str, width: int = 60) -> str:
    grouped = " ".join(sequence[i : i + 10] for i in range(0, len(sequence), 10))
    if len(grouped) > width:
        grouped = grouped[:width].rstrip() + " …"
    return grouped


def load_json_if_exists(path_str: str) -> Any | None:
    path = resolve_path(path_str)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_predictions_if_exists(path_str: str) -> pd.DataFrame | None:
    path = resolve_path(path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def reliability_reports(summary: dict[str, Any]) -> list[dict[str, Any]]:
    recalibrated = summary.get("recalibrated") or {}
    reports = recalibrated.get("reports") or (summary.get("raw") or {}).get("reports") or []
    return [r for r in reports if r.get("rank") in _RANKS]


def draw_reliability_chart(reports: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=_SAND)
    ax.set_facecolor(_SAND)
    ax.plot([0, 1], [0, 1], linestyle="--", color=_NAVY, linewidth=1.0, alpha=0.35, label="perfect")
    for report in reports:
        rank = report["rank"]
        xs = [b["mean_confidence"] for b in report["bins"] if b.get("count", 0) > 0]
        ys = [b["accuracy"] for b in report["bins"] if b.get("count", 0) > 0]
        if not xs:
            continue
        ece = report.get("ece")
        label = f"{rank}" + (f"  ECE {ece:.3f}" if ece is not None else "")
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=_RANK_BAND[rank], label=label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence (bin mean)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability by taxonomic rank")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(_NAVY)
        spine.set_alpha(0.25)
    fig.tight_layout()
    return fig


def fcw_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return np.array(
        [false_confident_wrong_rate(y_true, y_pred, scores, float(t)) for t in thresholds],
        dtype=np.float64,
    )


def species_fcw_methods(heldout: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Species-level FCW ingredients for fallback vs always-answer."""
    y_true = heldout["true_species"].to_numpy()
    nearest = heldout["nearest_species"].to_numpy()
    scores = heldout["confidence_species"].to_numpy(dtype=np.float64)
    committed = heldout["predicted_rank"].to_numpy() == "species"
    y_fallback = np.array([n if c else None for n, c in zip(nearest, committed)], dtype=object)
    return {
        "hierarchical fallback": (y_true, y_fallback, scores),
        "always-answer (species)": (y_true, nearest, scores),
    }


def draw_fcw_chart(thresholds: np.ndarray, series: dict[str, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=_SAND)
    ax.set_facecolor(_SAND)
    colors = {
        "hierarchical fallback": _SEAFOAM,
        "always-answer (species)": _CORAL,
    }
    for name, ys in series.items():
        ax.plot(thresholds, ys, linewidth=2.0, color=colors.get(name, _TEAL), label=name)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("False-confident-wrong rate")
    ax.set_title("False-confident-wrong rate vs. threshold (held-out genera)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(_NAVY)
        spine.set_alpha(0.25)
    fig.tight_layout()
    return fig


def blast_lie_cases(heldout: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    """Held-out queries where a species-level identity call would name the wrong known species."""
    novel = heldout.copy()
    if "split" in novel.columns:
        novel = novel[novel["split"] == "test"]
    wrong = novel["nearest_species"] != novel["true_species"]
    confident = novel["confidence_species"] >= 0.5
    flagged = (
        novel["is_novel"].astype(bool)
        if "is_novel" in novel.columns
        else pd.Series(True, index=novel.index)
    )
    picked = novel[wrong & confident & flagged].sort_values("confidence_species", ascending=False)
    return picked.head(limit)


def _render_header(pipeline: Pipeline | None) -> None:
    meta = ""
    if pipeline is not None:
        meta = (
            f"{pipeline.n_train:,} reference sequences · {pipeline.n_species:,} species · "
            f"{pipeline.n_genera:,} genera · {pipeline.n_families:,} families · "
            f"calibrated on {pipeline.n_val:,}"
        )
    st.markdown(
        f"""
        <div class="record-header">
          <p class="record-kicker">eDNA specimen record</p>
          <h1>Identify this specimen</h1>
          <p>A ranked answer even when the read is new: species in the shallows,
          order in the deep. The cascade stops where the distance is no longer
          trustworthy — instead of guessing wrong.</p>
          <div class="record-meta">{html.escape(meta)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_classify(pipeline: Pipeline) -> tuple[FallbackPrediction | None, list[RelativeHit]]:
    raw_text = st.text_area(
        "Sequence",
        height=120,
        placeholder=">specimen\nACGTACGT…",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    lat_text = col1.text_input("Latitude (optional)", value="")
    lon_text = col2.text_input("Longitude (optional)", value="")

    if not st.button("Identify this specimen", type="primary"):
        return None, []

    try:
        sequence = clean_sequence_text(raw_text)
    except SequenceParseError as exc:
        st.error(str(exc))
        return None, []

    lat = float(lat_text) if lat_text.strip() else None
    lon = float(lon_text) if lon_text.strip() else None
    if (lat is None) != (lon is None):
        st.error("Provide both latitude and longitude, or leave both blank.")
        return None, []

    prediction = classify_sequence(pipeline, sequence, lat, lon)
    relatives = nearest_relatives(pipeline, sequence, lat, lon, k=5)

    st.markdown(
        f'<div class="specimen-seq">{html.escape(format_specimen_sequence(sequence))}</div>',
        unsafe_allow_html=True,
    )

    if prediction.is_novel:
        st.warning(
            f"Novel taxon — closest known relative: {prediction.closest_relative_species}."
        )
    else:
        st.success(f"Resolved at species: {prediction.species}")

    st.markdown(build_depth_tree_html(prediction), unsafe_allow_html=True)
    return prediction, relatives


def _render_relatives(prediction: FallbackPrediction, relatives: list[RelativeHit]) -> None:
    if not prediction.is_novel or not relatives:
        return
    st.markdown('<p class="panel-kicker">Nearest relatives</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Top known species by the same distance the cascade '
        "used. Shown because this read was flagged novel.</p>",
        unsafe_allow_html=True,
    )
    for hit in relatives:
        lineage = " · ".join(part for part in (hit.genus, hit.family, hit.order) if part)
        st.markdown(
            f'<div class="relative-row"><span class="name">{html.escape(hit.species)}</span>'
            f'<div class="mono-quiet">d={hit.distance:.3f} · n={hit.support}'
            f'{" · " + html.escape(lineage) if lineage else ""}</div></div>',
            unsafe_allow_html=True,
        )


def _render_calibration(summary: dict[str, Any] | None) -> None:
    if summary is None:
        return
    reports = reliability_reports(summary)
    if not reports:
        return
    st.markdown('<p class="panel-kicker">Calibration</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Does reported confidence mean what it says? '
        "Each rank is the same depth color as the tree. Source: held-out genera, "
        "reliability bins from validate_calibration.</p>",
        unsafe_allow_html=True,
    )
    fig = draw_reliability_chart(reports)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_blast_gallery(heldout: pd.DataFrame | None) -> None:
    if heldout is None or heldout.empty:
        return
    cases = blast_lie_cases(heldout)
    if cases.empty:
        return
    st.markdown('<p class="panel-kicker">BLAST would have lied to you</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Held-out genera: a species-level identity call '
        "(BLAST’s failure mode) names the nearest known species with high "
        "confidence. Coral marks the wrong confident call. We flagged novelty "
        "instead.</p>",
        unsafe_allow_html=True,
    )
    cards: list[str] = []
    for _, row in cases.iterrows():
        rank = row.get("predicted_rank") or "unresolved"
        cards.append(
            '<div class="lie-card">'
            f'<div class="coral-flag">wrong species call</div>'
            f'<div class="name">{html.escape(str(row["nearest_species"]))}</div>'
            f'<div class="mono-quiet">species confidence {float(row["confidence_species"]) * 100:.0f}%</div>'
            f'<div class="mono-quiet" style="margin-top:0.4rem">true · {html.escape(str(row["true_species"]))}</div>'
            f'<div class="mono-quiet">ours · {html.escape(str(rank))}</div>'
            "</div>"
        )
    st.markdown(f'<div class="lie-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_fcw(heldout: pd.DataFrame | None) -> None:
    if heldout is None or heldout.empty:
        return
    data = heldout
    if "split" in data.columns:
        data = data[data["split"] == "test"]
    if data.empty or "true_species" not in data.columns:
        return

    st.markdown('<p class="panel-kicker">Benchmark · false-confident-wrong</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Rate of wrong species names issued at or above a '
        "confidence threshold, on held-out genera. Toggle the always-answer "
        "curve to see the BLAST / Naive-Bayes failure mode against the cascade.</p>",
        unsafe_allow_html=True,
    )
    methods = species_fcw_methods(data)
    selected = st.multiselect(
        "Methods",
        options=list(methods.keys()),
        default=list(methods.keys()),
        label_visibility="collapsed",
    )
    if not selected:
        return
    thresholds = np.linspace(0.0, 1.0, 21)
    series = {
        name: fcw_series(y_true, y_pred, scores, thresholds)
        for name in selected
        for y_true, y_pred, scores in [methods[name]]
    }
    fig = draw_fcw_chart(thresholds, series)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_benchmark_table(results: list[dict[str, Any]] | None) -> None:
    if not results:
        return
    st.markdown(
        '<p class="panel-lede">Coverage / accuracy-among-answered on clade-exclusion '
        "splits (point estimate [bootstrap CI] when present).</p>",
        unsafe_allow_html=True,
    )
    for split_result in results:
        pct = round(split_result["holdout_fraction"] * 100)
        st.markdown(
            f'<p class="mono-quiet">{pct}% genus holdout · '
            f'{split_result["n_genera_held_out"]}/{split_result["n_genera_total"]} genera · '
            f'n_test={split_result["n_test"]:,}</p>',
            unsafe_allow_html=True,
        )
        rows = []
        for system_name, scores in split_result["systems"].items():
            if scores is None:
                rows.append({"system": system_name, **{f"{r}_coverage": None for r in _RANKS}})
                continue
            row: dict[str, Any] = {"system": system_name}
            for rank in _RANKS:
                row[f"{rank}_coverage"] = _format_metric_with_ci(
                    scores[rank].get("coverage"), scores[rank].get("coverage_ci")
                )
                row[f"{rank}_accuracy"] = _format_metric_with_ci(
                    scores[rank].get("accuracy_among_answered"),
                    scores[rank].get("accuracy_among_answered_ci"),
                )
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("system"), width="stretch")


def _format_metric_with_ci(value: float | None, ci: list | None) -> str | None:
    if value is None:
        return None
    text = f"{value:.3f}"
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        text += f" [{ci[0]:.3f}-{ci[1]:.3f}]"
    return text


def main() -> None:
    st.set_page_config(page_title="eDNA specimen record", layout="centered")
    _inject_theme()

    config = load_app_config()
    pipeline = _cached_pipeline("configs/app.yaml")
    _render_header(pipeline)

    prediction: FallbackPrediction | None = None
    relatives: list[RelativeHit] = []
    if pipeline is None:
        st.warning(
            "No reference data found at "
            f"`{config['data']['sequences_path']}`. Run "
            "`python -m src.ingest.fetch_bold` and `python -m src.preprocess.clean` first."
        )
    else:
        prediction, relatives = _render_classify(pipeline)

    calibration_path = config.get(
        "calibration_summary_path", "data/eval_results/calibration/calibration.json"
    )
    predictions_path = config.get(
        "calibration_predictions_path",
        "data/eval_results/calibration/heldout_predictions.parquet",
    )
    benchmark_path = config.get("benchmark_results_path", "data/eval_results/benchmark.json")

    summary = load_json_if_exists(calibration_path)
    heldout = load_predictions_if_exists(predictions_path)
    benchmark = load_json_if_exists(benchmark_path)

    show_relatives = prediction is not None and prediction.is_novel and bool(relatives)
    if show_relatives:
        cal_col, rel_col = st.columns(2)
        with cal_col:
            _render_calibration(summary)
        with rel_col:
            assert prediction is not None
            _render_relatives(prediction, relatives)
    else:
        _render_calibration(summary)

    _render_blast_gallery(heldout)
    _render_fcw(heldout)
    _render_benchmark_table(benchmark)


if __name__ == "__main__":
    main()
