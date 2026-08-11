"""Streamlit dashboard (Step 7 of the build plan).

Two tabs:
- **Classify**: paste/upload a DNA sequence (raw or FASTA), optionally give
  a lat/lon, and see the hierarchical, novelty-aware prediction --
  resolved rank, per-rank confidence, and (when flagged novel) the top-k
  nearest known species and genera by distance.
- **Benchmark Results**: renders the Step 6 comparison (``data/eval_results/
  benchmark.json``) -- our system vs. the Naive Bayes / 1-NN / BLAST
  baselines, per-rank coverage and accuracy-among-answered.

Run with ``streamlit run app/dashboard.py`` from the repo root (or see
``Dockerfile.inference``). Requires ``data/processed/sequences.parquet``
to already exist -- run ``python -m src.ingest.fetch_bold`` and
``python -m src.preprocess.clean`` first if it doesn't.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `streamlit run app/dashboard.py`

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

_RANKS = ("species", "genus", "family", "order")
_TREE_RANKS = ("order", "family", "genus", "species")  # coarse -> fine, for drawing top-down

# Confidence color thresholds mirror HierarchicalFallback's own semantics: a
# node's fill color is a direct visual read of prediction.confidence[rank],
# not a separate judgment call.
_CONFIDENT_COLOR = "#1b7a3d"
_BORDERLINE_COLOR = "#c98a12"
_LOW_CONFIDENCE_COLOR = "#b3261e"
_NOVEL_FILL_COLOR = "#f1f3f4"
_NOVEL_FONT_COLOR = "#3c4043"
_NOVEL_BORDER_COLOR = "#9aa0a6"


@st.cache_resource(show_spinner="Fitting encoder + classifier + fallback on the known reference data...")
def _cached_pipeline(config_path: str) -> Pipeline | None:
    config = load_app_config(config_path)
    try:
        return load_pipeline(config)
    except FileNotFoundError:
        return None


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.66:
        return _CONFIDENT_COLOR
    if confidence >= 0.33:
        return _BORDERLINE_COLOR
    return _LOW_CONFIDENCE_COLOR


def _escape_dot_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _format_metric_with_ci(value: float | None, ci: list | None) -> str | None:
    if value is None:
        return None
    text = f"{value:.3f}"
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        text += f" [{ci[0]:.3f}-{ci[1]:.3f}]"
    return text


def _build_taxonomy_tree_dot(prediction: FallbackPrediction) -> str:
    """Renders the order->family->genus->species cascade as a DOT graph.

    Each resolved node's fill color is that rank's calibrated confidence
    (green/amber/red); a dashed leaf marks exactly where the cascade
    dropped below threshold, labeled with the nearest known species --
    the same top hit the model reports, with the full top-k list shown
    beside the tree so the claim is inspectable.
    """
    lines = [
        "digraph taxonomy {",
        'graph [rankdir=LR, bgcolor="transparent", nodesep=0.5, ranksep=0.7];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=13, fontcolor="white", color="#ffffff33", penwidth=1];',
        'edge [fontname="Helvetica", fontsize=10, color="#9aa0a6", arrowsize=0.8];',
    ]

    previous_node_id: str | None = None
    for rank in _TREE_RANKS:
        value = getattr(prediction, rank)
        if value is None:
            break
        node_id = f"n_{rank}"
        confidence = prediction.confidence.get(rank, 0.0)
        label = _escape_dot_label(f"{value}\n{rank} \u00b7 {confidence * 100:.0f}%")
        lines.append(f'{node_id} [label="{label}", fillcolor="{_confidence_color(confidence)}"];')
        if previous_node_id is not None:
            lines.append(f"{previous_node_id} -> {node_id};")
        previous_node_id = node_id

    if prediction.is_novel:
        novel_label = _escape_dot_label(
            f"novel taxon\nnearest species:\n{prediction.closest_relative_species}"
        )
        lines.append(
            f'n_novel [label="{novel_label}", style="rounded,dashed,filled", '
            f'fillcolor="{_NOVEL_FILL_COLOR}", fontcolor="{_NOVEL_FONT_COLOR}", color="{_NOVEL_BORDER_COLOR}"];'
        )
        if previous_node_id is not None:
            lines.append(f'{previous_node_id} -> n_novel [style=dashed, label="below threshold"];')

    lines.append("}")
    return "\n".join(lines)


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
    """Species then genera, each already ordered by distance at that rank.

    Distances are not comparable across ranks (different centroid scales),
    so this is two concatenated lists, not a single mixed ranking.
    """
    return pd.DataFrame(_hits_to_rows(prediction.nearest_species) + _hits_to_rows(prediction.nearest_genera))


def _render_classify_tab(pipeline: Pipeline) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Training sequences", f"{pipeline.n_train:,}")
    metric_cols[1].metric("Species", f"{pipeline.n_species:,}")
    metric_cols[2].metric("Genera", f"{pipeline.n_genera:,}")
    metric_cols[3].metric("Families / Orders", f"{pipeline.n_families:,} / {pipeline.n_orders:,}")
    st.caption(f"Calibrated on {pipeline.n_val:,} held-back validation sequences.")

    raw_text = st.text_area(
        "Paste a DNA sequence (raw bases or a full FASTA record)",
        height=160,
        placeholder=">my_read\nACGTACGT...",
    )

    col1, col2 = st.columns(2)
    lat_text = col1.text_input("Latitude (optional)", value="")
    lon_text = col2.text_input("Longitude (optional)", value="")

    if not st.button("Classify", type="primary"):
        return

    try:
        sequence = clean_sequence_text(raw_text)
    except SequenceParseError as exc:
        st.error(str(exc))
        return

    lat = float(lat_text) if lat_text.strip() else None
    lon = float(lon_text) if lon_text.strip() else None
    if (lat is None) != (lon is None):
        st.error("Provide both latitude and longitude, or leave both blank.")
        return

    prediction = classify_sequence(pipeline, sequence, lat, lon)

    st.divider()

    if prediction.is_novel:
        top_species = ", ".join(hit.label for hit in prediction.nearest_species[:3]) or prediction.closest_relative_species
        st.warning(
            f"**Novel taxon** (no confident call at species or genus level). "
            f"Nearest known species: *{prediction.closest_relative_species}* "
            f"(top hits: {top_species}). Inspect the ranked neighbors below — "
            f"this is not a single black-box name."
        )
    else:
        st.success(f"**{prediction.species}** (resolved at species level)")

    tree_col, confidence_col = st.columns([3, 2])

    with tree_col:
        st.subheader("Taxonomic tree")
        st.graphviz_chart(_build_taxonomy_tree_dot(prediction), width="stretch")
        st.caption(
            "Node color = that rank's calibrated confidence (green = confident, "
            "amber = borderline, red = low). Dashed branch = below threshold, i.e. "
            "the point the model chose to flag novelty instead of guessing."
        )

    with confidence_col:
        st.subheader("Per-rank confidence")
        confidence_df = pd.DataFrame(
            {
                "Rank": [r.capitalize() for r in _RANKS],
                "Confidence": [prediction.confidence[r] * 100 for r in _RANKS],
                "Distance": [prediction.distance[r] for r in _RANKS],
                "Support (n)": [prediction.support[r] for r in _RANKS],
            }
        )
        st.dataframe(
            confidence_df,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Distance confidence × n/(n+prior); thinly sampled centroids are deflated.",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.0f%%",
                ),
                "Distance": st.column_config.NumberColumn("Distance (scaled)", format="%.3f"),
                "Support (n)": st.column_config.NumberColumn(
                    "Support (n)", help="Training rows backing the nearest label at this rank."
                ),
            },
            hide_index=True,
            width="stretch",
        )

    if prediction.is_novel:
        st.subheader("Nearest known relatives")
        st.caption(
            "Top known species and genera by the same combined distance the "
            "fallback uses. Species and genera are listed separately: distances "
            "are scaled per rank and are not comparable across ranks."
        )
        relatives_df = nearest_relatives_dataframe(prediction)
        st.dataframe(
            relatives_df,
            column_config={
                "Distance": st.column_config.NumberColumn("Distance (scaled)", format="%.3f"),
                "Support (n)": st.column_config.NumberColumn(
                    "Support (n)", help="Training rows backing this centroid."
                ),
            },
            hide_index=True,
            width="stretch",
        )


def _render_benchmark_tab(benchmark_results_path: str) -> None:
    path = resolve_path(benchmark_results_path)
    if not path.exists():
        st.info(
            "No benchmark results found yet. Run "
            "`python -m src.eval.benchmark --config configs/eval.yaml` to generate "
            f"`{benchmark_results_path}`."
        )
        return

    with open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)

    for split_result in results:
        pct = round(split_result["holdout_fraction"] * 100)
        st.subheader(
            f"{pct}% genus holdout "
            f"({split_result['n_genera_held_out']}/{split_result['n_genera_total']} genera, "
            f"n_test={split_result['n_test']:,})"
        )

        rows = []
        for system_name, scores in split_result["systems"].items():
            if scores is None:
                rows.append({"system": system_name, **{f"{r}_coverage": None for r in _RANKS}})
                continue
            row = {"system": system_name}
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
        st.caption("Values are point estimate [bootstrap CI] when CIs are present in benchmark.json.")

        accuracy_by_rank = pd.DataFrame(
            {
                system_name: [scores[rank]["accuracy_among_answered"] for rank in _RANKS]
                for system_name, scores in split_result["systems"].items()
                if scores is not None
            },
            index=[rank.capitalize() for rank in _RANKS],
        )
        st.caption("Accuracy among answered, by rank (higher is better; missing bars = system skipped)")
        st.bar_chart(accuracy_by_rank, width="stretch")


def main() -> None:
    st.set_page_config(page_title="eDNA Biodiversity Classifier", layout="wide")
    st.title("eDNA Biodiversity Classifier")
    st.caption(
        "Instead of failing outright when a read doesn't match any known reference "
        "sequence, this system always returns a usable, ranked answer -- falling back "
        "to genus/family/order and flagging genuinely novel taxa rather than guessing wrong."
    )
    st.divider()

    config = load_app_config()
    pipeline = _cached_pipeline("configs/app.yaml")

    classify_tab, benchmark_tab = st.tabs(["Classify", "Benchmark Results"])

    with classify_tab:
        if pipeline is None:
            st.warning(
                "No reference data found at "
                f"`{config['data']['sequences_path']}`. Run "
                "`python -m src.ingest.fetch_bold` and `python -m src.preprocess.clean` first."
            )
        else:
            _render_classify_tab(pipeline)

    with benchmark_tab:
        _render_benchmark_tab(config["benchmark_results_path"])


if __name__ == "__main__":
    main()
