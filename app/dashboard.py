"""Streamlit dashboard (Step 7 of the build plan, plus Step 4 evidence panels).

Tabs:
- **Classify**: paste a DNA sequence and see the hierarchical, novelty-aware
  prediction (resolved rank, per-rank confidence, top-k nearest relatives).
- **Benchmark Results**: Step 6 coverage / accuracy-among-answered table.
- **Calibration**: live reliability diagram from Step 1 ``calibration.json``
  (optional re-bin of ``heldout_predictions.parquet``).
- **BLAST would have lied**: 3–5 hardcoded held-out contrasts (Step 2/3).
- **False-confident wrong**: interactive FCW vs. threshold from Step 2.

Run with ``streamlit run app/dashboard.py`` from the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `streamlit run app/dashboard.py`

from app.dashboard_charts import (  # noqa: E402
    confidence_gradient_css,
    fcw_at_threshold,
    fcw_curve_frame,
    format_sequence_html,
    live_reliability_reports,
    reliability_frame,
    reports_to_dicts,
)
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

_RANKS = ("species", "genus", "family", "order")
_TREE_RANKS = ("order", "family", "genus", "species")  # coarse -> fine, for drawing top-down
_METHOD_OPTIONS = ("ours", "blast", "naive_bayes", "nearest_neighbor")
_METHOD_LABELS = {
    "ours": "ours",
    "blast": "BLAST",
    "naive_bayes": "Naive Bayes",
    "nearest_neighbor": "1-NN",
}

_CONFIDENT_COLOR = "#1b7a3d"
_BORDERLINE_COLOR = "#c98a12"
_LOW_CONFIDENCE_COLOR = "#b3261e"
_NOVEL_FILL_COLOR = "#f1f3f4"
_NOVEL_FONT_COLOR = "#3c4043"
_NOVEL_BORDER_COLOR = "#9aa0a6"

_PAGE_CSS = """
<style>
@keyframes edna-rank-reveal {
  from { opacity: 0; transform: translateY(12px) scale(0.97); }
  to { opacity: 1; transform: none; }
}
.edna-rank-cascade { display: flex; gap: 0.55rem; flex-wrap: wrap; margin: 0.4rem 0 0.8rem; }
.edna-rank-card {
  animation: edna-rank-reveal 0.55s ease both;
  border-radius: 12px;
  padding: 0.65rem 0.85rem;
  color: #fff;
  min-width: 8.5rem;
  box-shadow: 0 10px 22px rgba(0,0,0,0.14);
}
.edna-rank-card.novel {
  color: #3c4043;
  border: 1.5px dashed #9aa0a6;
}
.edna-rank-name { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.9; }
.edna-rank-value { font-size: 1.02rem; font-weight: 650; margin: 0.15rem 0; }
.edna-rank-conf { font-size: 0.78rem; font-variant-numeric: tabular-nums; }
.edna-gallery-wrong { background: #fdecea; border: 1px solid #f5c6c2; border-radius: 12px; padding: 0.9rem 1rem; }
.edna-gallery-honest { background: #e8f5ee; border: 1px solid #b7e0c6; border-radius: 12px; padding: 0.9rem 1rem; }
</style>
"""


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


def _rank_reveal_html(prediction: FallbackPrediction) -> str:
    cards: list[str] = []
    for i, rank in enumerate(_TREE_RANKS):
        value = getattr(prediction, rank)
        delay = f"{0.12 * i:.2f}s"
        if value is None:
            cards.append(
                f'<div class="edna-rank-card novel" style="animation-delay:{delay};'
                f'background:{_NOVEL_FILL_COLOR};">'
                f'<div class="edna-rank-name">{rank}</div>'
                f'<div class="edna-rank-value">unresolved</div>'
                f'<div class="edna-rank-conf">below threshold</div></div>'
            )
            continue
        conf = float(prediction.confidence.get(rank, 0.0))
        bg = confidence_gradient_css(conf)
        cards.append(
            f'<div class="edna-rank-card" style="animation-delay:{delay};background:{bg};">'
            f'<div class="edna-rank-name">{rank}</div>'
            f'<div class="edna-rank-value">{value}</div>'
            f'<div class="edna-rank-conf">{conf * 100:.0f}% confidence</div></div>'
        )
    return f'<div class="edna-rank-cascade">{"".join(cards)}</div>'


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
    st.markdown(format_sequence_html(sequence), unsafe_allow_html=True)
    st.caption(f"{len(sequence)} bp · base colors A/C/G/T")

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

    st.markdown(_rank_reveal_html(prediction), unsafe_allow_html=True)
    st.caption("Cards fill on a confidence gradient (red → amber → green) and reveal coarse-to-fine.")

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


def _altair_reliability(frame: pd.DataFrame):
    import altair as alt

    perfect = pd.DataFrame({"mean_confidence": [0.0, 1.0], "accuracy": [0.0, 1.0]})
    diagonal = (
        alt.Chart(perfect)
        .mark_line(strokeDash=[5, 4], color="#9aa0a6")
        .encode(x="mean_confidence:Q", y="accuracy:Q")
    )
    if frame.empty:
        return diagonal.properties(title="No occupied confidence bins")
    points = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("mean_confidence:Q", title="Predicted confidence (bin mean)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("accuracy:Q", title="Observed accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("rank:N", title="Rank"),
            tooltip=["rank", "mean_confidence", "accuracy", "count", "ece"],
        )
    )
    return (diagonal + points).properties(height=420)


def _render_calibration_tab(calibration_results_path: str) -> None:
    st.caption(
        "Step 1 reliability diagram: bin held-out confidence vs. whether the nearest "
        "label at that rank was correct. ECE is the size-weighted |acc − conf|."
    )
    path = resolve_path(calibration_results_path)
    if not path.exists():
        st.info(
            "No calibration results yet. Run "
            "`python -m src.eval.validate_calibration --config configs/eval.yaml` "
            f"to write `{calibration_results_path}`."
        )
        return

    with open(path, "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    source = st.radio(
        "Curve source",
        options=["raw", "recalibrated"],
        horizontal=True,
        help="Recalibrated is leave-one-holdout-out isotonic/Platt on ranks whose raw ECE failed the gate.",
    )
    block = summary.get(source) or summary.get("raw")
    if not block:
        st.warning("calibration.json has no reliability reports.")
        return

    predictions_path = summary.get("predictions_path")
    predictions_file = resolve_path(predictions_path) if predictions_path else None
    n_bins = int(summary.get("n_bins", 10))
    if predictions_file is not None and predictions_file.exists():
        n_bins = int(st.slider("Confidence bins (live re-bin of held-out predictions)", 4, 20, n_bins))
        pred_df = pd.read_parquet(predictions_file)
        if "split" in pred_df.columns:
            pred_df = pred_df[pred_df["split"] == "test"]
        reports = reports_to_dicts(live_reliability_reports(pred_df, n_bins=n_bins))
        st.caption(f"Live re-bin of {len(pred_df):,} held-out test rows from `{predictions_file.name}`.")
    else:
        reports = block.get("reports") or []
        st.caption("Live re-bin needs `heldout_predictions.parquet` beside calibration.json; showing saved bins.")

    ece_cols = st.columns(len(_RANKS))
    ece_by_rank = {r["rank"]: r.get("ece") for r in reports}
    for i, rank in enumerate(_RANKS):
        ece = ece_by_rank.get(rank)
        ece_cols[i].metric(f"{rank} ECE", "—" if ece is None else f"{ece:.3f}")

    frame = reliability_frame(reports)
    st.altair_chart(_altair_reliability(frame), width="stretch")
    st.caption(f"Decision from validate_calibration: **{summary.get('decision', 'unknown')}**.")


def _render_gallery_example(example: BlastLieExample) -> None:
    st.markdown(f"**{example.true_species}** · held-out genus *{example.true_genus}* ({example.holdout_fraction:.0%} split)")
    st.markdown(format_sequence_html(example.sequence), unsafe_allow_html=True)

    blast_col, ours_col = st.columns(2)
    with blast_col:
        st.markdown(
            f'<div class="edna-gallery-wrong"><strong>BLAST would have lied</strong><br/>'
            f"Called <em>{example.blast_species}</em> at {example.blast_pident:.1f}% identity "
            f"(bitscore {example.blast_bitscore:.0f}, confidence {example.blast_confidence:.2f}). "
            f"Wrong species and wrong genus — the hit is a seen sister, not the held-out taxon."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write(
            {
                "species": example.blast_species,
                "genus": example.blast_genus,
                "family": example.blast_family,
                "order": example.blast_order,
            }
        )
    with ours_col:
        resolved = example.ours_predicted_rank or "abstain"
        st.markdown(
            f'<div class="edna-gallery-honest"><strong>Honest fallback</strong><br/>'
            f"Flags novel; commits only at <em>{resolved}</em> "
            f"({example.ours_family or example.ours_order or 'no rank'}). "
            f"Nearest known relatives are listed, not a false species name."
            f"</div>",
            unsafe_allow_html=True,
        )
        relatives = pd.DataFrame(
            [
                {"list": "species", "label": hit.label, "distance": hit.distance, "genus": hit.genus}
                for hit in example.nearest_species
            ]
            + [
                {"list": "genus", "label": hit.label, "distance": hit.distance, "genus": hit.genus}
                for hit in example.nearest_genera
            ]
        )
        st.dataframe(relatives, hide_index=True, width="stretch")


def _render_gallery_tab() -> None:
    st.caption(
        "Hardcoded 50% genus-holdout cases: BLAST’s operating point (≥97% identity) "
        "still copies a seen sister-genus name with high bitscore. Ours refuses the "
        "species/genus call and shows the Step 3 nearest-relative list instead."
    )
    for example in BLAST_LIE_EXAMPLES:
        with st.expander(f"{example.true_species}  vs.  BLAST: {example.blast_species}", expanded=example is BLAST_LIE_EXAMPLES[0]):
            _render_gallery_example(example)


def _altair_fcw(frame: pd.DataFrame, threshold: float):
    import altair as alt

    if frame.empty:
        return alt.Chart(pd.DataFrame({"threshold": [0, 1], "fcw": [0, 0]})).mark_line().encode(
            x="threshold:Q", y="fcw:Q"
        )
    lines = (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("threshold:Q", title="Confidence / score threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("fcw:Q", title="False-confident-wrong-call rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("label:N", title="Method"),
            tooltip=["label", "threshold", "fcw"],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#3c4043", strokeDash=[4, 3])
        .encode(x="threshold:Q")
    )
    return (lines + rule).properties(height=420)


def _render_fcw_tab(head_to_head_results_path: str) -> None:
    st.caption(
        "Step 2 metric: fraction of queries that are a wrong *call* with score ≥ threshold. "
        "Toggle methods and slide the threshold; the published curve is the 50% holdout."
    )
    path = resolve_path(head_to_head_results_path)
    if not path.exists():
        st.info(
            "No head-to-head results yet. Run "
            "`python -m src.eval.head_to_head --config configs/eval.yaml` "
            f"to write `{head_to_head_results_path}`."
        )
        return

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    available = []
    for split in payload.get("splits") or []:
        for method, scores in (split.get("systems") or {}).items():
            if scores is not None:
                available.append(method)
    available = tuple(dict.fromkeys(available)) or _METHOD_OPTIONS

    methods = st.multiselect(
        "Methods",
        options=list(available),
        default=list(available),
        format_func=lambda m: _METHOD_LABELS.get(m, m),
    )
    rank = st.selectbox("Rank", list(_RANKS), index=0)
    subset = st.radio("Subset", options=["novel", "all"], horizontal=True, format_func=lambda s: "held-out genera" if s == "novel" else "val+test")
    default_t = float(payload.get("default_confidence_threshold", 0.5))
    threshold = st.slider("Confidence threshold", 0.0, 1.0, default_t, 0.05)

    if not methods:
        st.warning("Select at least one method.")
        return

    curve_df = fcw_curve_frame(payload, rank=rank, subset=subset, methods=methods)
    at_t = fcw_at_threshold(curve_df, threshold)
    if not at_t.empty:
        metric_cols = st.columns(max(len(at_t), 1))
        for i, row in at_t.iterrows():
            metric_cols[int(i) % len(metric_cols)].metric(str(row["label"]), f"{row['fcw']:.3f}")

    st.altair_chart(_altair_fcw(curve_df, threshold), width="stretch")
    n_points = len((payload.get("splits") or [{}])[0].get("thresholds") or [])
    st.caption(
        f"Holdout {float(payload.get('table_holdout_fraction', 0.5)):.0%} · "
        f"values interpolate the saved {n_points}-point sweep."
    )


def main() -> None:
    st.set_page_config(page_title="eDNA Biodiversity Classifier", layout="wide")
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    st.title("eDNA Biodiversity Classifier")
    st.caption(
        "Instead of failing outright when a read doesn't match any known reference "
        "sequence, this system always returns a usable, ranked answer -- falling back "
        "to genus/family/order and flagging genuinely novel taxa rather than guessing wrong."
    )
    st.divider()

    config = load_app_config()
    pipeline = _cached_pipeline("configs/app.yaml")

    classify_tab, benchmark_tab, calibration_tab, gallery_tab, fcw_tab = st.tabs(
        [
            "Classify",
            "Benchmark Results",
            "Calibration",
            "BLAST would have lied",
            "False-confident wrong",
        ]
    )

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

    with calibration_tab:
        _render_calibration_tab(config.get("calibration_results_path", "data/eval_results/calibration/calibration.json"))

    with gallery_tab:
        _render_gallery_tab()

    with fcw_tab:
        _render_fcw_tab(config.get("head_to_head_results_path", "data/eval_results/head_to_head/head_to_head.json"))


if __name__ == "__main__":
    main()
