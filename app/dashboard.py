"""Streamlit dashboard: a specimen record, not a BI layout.

Identify / Evidence / Method tabs keep the cascade call separate from the
proof and the benchmark. Color is taxonomic confidence as ocean depth --
species (trustworthy) is shallow seafoam, order (the last fallback) is
deep navy -- the same cascade ``HierarchicalFallback`` uses. Only the
depth-tree reveal animates; calibration, BLAST-lie gallery, and
false-confident-wrong charts stay quiet underneath.
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

from app.dashboard_charts import (  # noqa: E402
    benchmark_bar_frame,
    build_benchmark_bar_figure,
    fcw_at_threshold,
    fcw_callout_row,
    fcw_curve_frame,
    holdout_percent_options,
    live_reliability_reports,
    reliability_frame,
    reports_to_dicts,
    split_at_holdout,
)
from app.depth_tree import (  # noqa: E402,F401
    build_depth_tree_html,
    rank_visual_state,
    render_depth_tree,
)
from app.family_icons import result_callout_html  # noqa: E402
from app.species_image import render_reference_photo  # noqa: E402
from app.gallery_examples import BLAST_LIE_EXAMPLES, BlastLieExample  # noqa: E402
from app.pipeline import (  # noqa: E402
    Pipeline,
    SequenceParseError,
    classify_sequence,
    clean_sequence_text,
    load_app_config,
    load_pipeline,
)
from src.common import resolve_path  # noqa: E402
from src.fallback.novelty import FallbackPrediction, NeighborHit  # noqa: E402
from theme.inject import inject_theme  # noqa: E402
from theme.tokens import CORAL, NAVY, RANK_BAND, SAND, SEAFOAM, TEAL  # noqa: E402

_RANKS = ("species", "genus", "family", "order")
_TAB_LABELS = ("Identify", "Evidence", "Method")
_PRED_STATE = "identify_prediction"
_SEQ_STATE = "identify_sequence"
_METHOD_OPTIONS = ("ours", "blast", "naive_bayes", "nearest_neighbor")
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


@st.cache_resource(show_spinner="Fitting encoder + classifier + fallback on the known reference data...")
def _cached_pipeline(config_path: str) -> Pipeline | None:
    config = load_app_config(config_path)
    try:
        return load_pipeline(config)
    except FileNotFoundError:
        return None


def format_specimen_sequence(sequence: str, width: int = 60) -> str:
    grouped = " ".join(sequence[i : i + 10] for i in range(0, len(sequence), 10))
    if len(grouped) > width:
        grouped = grouped[:width].rstrip() + " …"
    return grouped


def _hits_to_rows(hits: list[NeighborHit]) -> list[dict[str, object]]:
    rows = []
    for i, hit in enumerate(hits, start=1):
        rows.append(
            {
                "Rank": i,
                "Taxon rank": hit.rank.capitalize(),
                "Label": hit.label,
                "Genus": hit.genus,
                "Family": hit.family,
                "Distance": hit.distance,
                "Support (n)": hit.support,
            }
        )
    return rows


def nearest_relatives_dataframe(prediction: FallbackPrediction) -> pd.DataFrame:
    """Species then genera, each already ordered by distance at that rank."""
    return pd.DataFrame(_hits_to_rows(prediction.nearest_species) + _hits_to_rows(prediction.nearest_genera))


def load_json_if_exists(path_str: str) -> Any | None:
    path = resolve_path(path_str)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def reliability_reports(summary: dict[str, Any]) -> list[dict[str, Any]]:
    recalibrated = summary.get("recalibrated") or {}
    reports = recalibrated.get("reports") or (summary.get("raw") or {}).get("reports") or []
    return [r for r in reports if r.get("rank") in _RANKS]


def draw_reliability_chart(frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SAND)
    ax.set_facecolor(SAND)
    ax.plot([0, 1], [0, 1], linestyle="--", color=NAVY, linewidth=1.0, alpha=0.35, label="perfect")
    for rank, group in frame.groupby("rank", sort=False):
        ece = group["ece"].iloc[0] if "ece" in group.columns else float("nan")
        label = f"{rank}" + (f"  ECE {ece:.3f}" if np.isfinite(ece) else "")
        ax.plot(
            group["mean_confidence"],
            group["accuracy"],
            marker="o",
            linewidth=1.8,
            color=RANK_BAND.get(str(rank), TEAL),
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence (bin mean)")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability by taxonomic rank")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(NAVY)
        spine.set_alpha(0.25)
    fig.tight_layout()
    return fig


def draw_fcw_chart(curve_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SAND)
    ax.set_facecolor(SAND)
    for method, group in curve_df.groupby("method", sort=False):
        ax.plot(
            group["threshold"],
            group["fcw"],
            linewidth=2.0,
            color=_METHOD_COLOR.get(str(method), TEAL),
            label=group["label"].iloc[0],
        )
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("False-confident-wrong rate")
    ax.set_title("False-confident-wrong rate vs. threshold (held-out genera)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(NAVY)
        spine.set_alpha(0.25)
    fig.tight_layout()
    return fig


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


def _render_classify(pipeline: Pipeline) -> FallbackPrediction | None:
    raw_text = st.text_area(
        "Sequence",
        height=120,
        placeholder=">specimen\nACGTACGT…",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns(2)
    lat_text = col1.text_input("Latitude (optional)", value="")
    lon_text = col2.text_input("Longitude (optional)", value="")

    # Tabs rerun the script; keep the last call so switching away from
    # Identify does not wipe the tree.
    if st.button("Identify this specimen", type="primary"):
        try:
            sequence = clean_sequence_text(raw_text)
        except SequenceParseError as exc:
            st.error(str(exc))
        else:
            lat = float(lat_text) if lat_text.strip() else None
            lon = float(lon_text) if lon_text.strip() else None
            if (lat is None) != (lon is None):
                st.error("Provide both latitude and longitude, or leave both blank.")
            else:
                st.session_state[_SEQ_STATE] = sequence
                st.session_state[_PRED_STATE] = classify_sequence(pipeline, sequence, lat, lon)

    prediction = st.session_state.get(_PRED_STATE)
    sequence = st.session_state.get(_SEQ_STATE)
    if prediction is None or sequence is None:
        return None

    st.markdown(
        f'<div class="specimen-seq">{html.escape(format_specimen_sequence(sequence))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(result_callout_html(prediction), unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-kicker">Taxonomic depth</p>'
        '<p class="panel-lede">Shallow water is a species-level call. '
        "Each node darker is one rank of fallback. The cascade stops at the "
        "last trustworthy rank; below that is unresolved water.</p>",
        unsafe_allow_html=True,
    )
    tree_col, photo_col = st.columns([1.35, 0.85], vertical_alignment="top")
    with tree_col:
        render_depth_tree(prediction, heading=False)
    with photo_col:
        render_reference_photo(prediction)
    return prediction


def _render_relatives(prediction: FallbackPrediction) -> None:
    if not prediction.is_novel:
        return
    st.markdown('<p class="panel-kicker">Nearest relatives</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Top known species and genera by the same distance '
        "the cascade used. Distances are per-rank and not comparable across ranks.</p>",
        unsafe_allow_html=True,
    )
    table = nearest_relatives_dataframe(prediction)
    if table.empty:
        return
    st.dataframe(
        table,
        column_config={
            "Distance": st.column_config.NumberColumn("Distance (scaled)", format="%.3f"),
            "Support (n)": st.column_config.NumberColumn("Support (n)"),
        },
        hide_index=True,
        width="stretch",
    )


def _render_calibration(config: dict[str, Any]) -> None:
    summary = load_json_if_exists(
        config.get("calibration_results_path", "data/eval_results/calibration/calibration.json")
    )
    predictions_path = resolve_path(
        config.get("calibration_predictions_path", "data/eval_results/calibration/heldout_predictions.parquet")
    )
    reports: list[dict[str, Any]] = []
    if predictions_path.exists():
        heldout = pd.read_parquet(predictions_path)
        n_bins = int((summary or {}).get("n_bins", 10))
        reports = reports_to_dicts(live_reliability_reports(heldout, n_bins=n_bins))
    elif summary is not None:
        reports = reliability_reports(summary)
    if not reports:
        return
    frame = reliability_frame(reports)
    if frame.empty:
        return
    st.markdown('<p class="panel-kicker">Calibration</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Does reported confidence mean what it says? '
        "Each rank uses the same depth color as the tree. Source: held-out genera.</p>",
        unsafe_allow_html=True,
    )
    fig = draw_reliability_chart(frame)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_blast_gallery() -> None:
    st.markdown('<p class="panel-kicker">BLAST would have lied to you</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Held-out genera: BLAST names a seen sister-genus '
        "species with high identity. Coral marks the wrong confident call. "
        "We flagged novelty and listed nearest known relatives instead.</p>",
        unsafe_allow_html=True,
    )
    for example in BLAST_LIE_EXAMPLES:
        _render_lie_example(example)


def _render_lie_example(example: BlastLieExample) -> None:
    rank = example.ours_predicted_rank or "unresolved"
    with st.expander(f"{example.true_species}  ·  BLAST named {example.blast_species}"):
        st.markdown(
            f'<div class="coral-flag">wrong species call</div>'
            f'<div class="name">{html.escape(example.blast_species)}</div>'
            f'<div class="mono-quiet">{example.blast_pident:.1f}% identity · '
            f'bitscore {example.blast_bitscore:.0f}</div>'
            f'<div class="mono-quiet" style="margin-top:0.4rem">true · '
            f'{html.escape(example.true_species)}</div>'
            f'<div class="mono-quiet">ours · {html.escape(str(rank))}</div>'
            f'<div class="specimen-seq">{html.escape(format_specimen_sequence(example.sequence))}</div>',
            unsafe_allow_html=True,
        )


def _render_fcw(config: dict[str, Any]) -> None:
    payload = load_json_if_exists(
        config.get("head_to_head_results_path", "data/eval_results/head_to_head/head_to_head.json")
    )
    if not payload:
        return
    st.markdown('<p class="panel-kicker">Benchmark · false-confident-wrong</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Wrong species names issued at or above a confidence '
        "threshold, on held-out genera. Toggle methods against the cascade.</p>",
        unsafe_allow_html=True,
    )
    selected = st.multiselect(
        "Methods",
        options=list(_METHOD_OPTIONS),
        default=["ours", "blast"],
        format_func=lambda key: _METHOD_LABELS[key],
        label_visibility="collapsed",
    )
    threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    if not selected:
        return
    curve = fcw_curve_frame(payload, rank="species", subset="novel", methods=selected)
    if curve.empty:
        return
    at_t = fcw_at_threshold(curve, threshold)
    headline = fcw_callout_row(at_t)
    if headline is not None:
        st.markdown(
            f'<div class="fcw-callout">'
            f'<div class="fcw-value">{headline["fcw"] * 100:.0f}%</div>'
            f'<p class="fcw-label">false-confident-wrong at {threshold:.0%} confidence · '
            f'{html.escape(str(headline["label"]))} · held-out genera</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    fig = draw_fcw_chart(curve)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_benchmark_chart(results: list[dict[str, Any]] | None) -> None:
    if not results:
        return
    options = holdout_percent_options(results)
    if not options:
        return
    st.markdown('<p class="panel-kicker">Benchmark · clade exclusion</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="panel-lede">Coverage and accuracy-among-answered on genus-holdout '
        "splits. Metric on the x-axis, method as color. Drag the holdout to restack.</p>",
        unsafe_allow_html=True,
    )
    default = 50 if 50 in options else options[0]
    percent = st.select_slider("Genus holdout", options=options, value=default, format_func=lambda pct: f"{pct}%")
    split = split_at_holdout(results, int(percent))
    if not split:
        return
    st.markdown(
        f'<p class="mono-quiet">{percent}% genus holdout · '
        f'{split["n_genera_held_out"]}/{split["n_genera_total"]} genera · '
        f'n_test={split["n_test"]:,}</p>',
        unsafe_allow_html=True,
    )
    frame = benchmark_bar_frame(split)
    if frame.empty:
        return
    fig = build_benchmark_bar_figure(frame)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def main() -> None:
    st.set_page_config(page_title="eDNA specimen record", layout="centered")
    inject_theme()

    config = load_app_config()
    pipeline = _cached_pipeline("configs/app.yaml")
    _render_header(pipeline)

    identify_tab, evidence_tab, method_tab = st.tabs(list(_TAB_LABELS))
    with identify_tab:
        prediction: FallbackPrediction | None = None
        if pipeline is None:
            st.warning(
                "No reference data found at "
                f"`{config['data']['sequences_path']}`. Run "
                "`python -m src.ingest.fetch_bold` and `python -m src.preprocess.clean` first."
            )
        else:
            prediction = _render_classify(pipeline)
        if prediction is not None:
            _render_relatives(prediction)
    with evidence_tab:
        _render_calibration(config)
        _render_blast_gallery()
    with method_tab:
        _render_fcw(config)
        _render_benchmark_chart(
            load_json_if_exists(config.get("benchmark_results_path", "data/eval_results/benchmark.json"))
        )


if __name__ == "__main__":
    main()
