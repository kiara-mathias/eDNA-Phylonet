"""Streamlit dashboard (Step 7 of the build plan).

Two tabs:
- **Classify**: paste/upload a DNA sequence (raw or FASTA), optionally give
  a lat/lon, and see the hierarchical, novelty-aware prediction --
  resolved rank, per-rank confidence, and (when flagged novel) the closest
  known relative.
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

_RANKS = ("species", "genus", "family", "order")


@st.cache_resource(show_spinner="Fitting encoder + classifier + fallback on the known reference data...")
def _cached_pipeline(config_path: str) -> Pipeline | None:
    config = load_app_config(config_path)
    try:
        return load_pipeline(config)
    except FileNotFoundError:
        return None


def _render_classify_tab(pipeline: Pipeline) -> None:
    st.caption(
        f"Fitted on {pipeline.n_train:,} training sequences "
        f"({pipeline.n_species:,} species / {pipeline.n_genera:,} genera / "
        f"{pipeline.n_families:,} families / {pipeline.n_orders:,} orders), "
        f"calibrated on {pipeline.n_val:,} held-back sequences."
    )

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

    if prediction.is_novel:
        st.warning(
            f"**Novel taxon** (no confident call at species or genus level). "
            f"Closest known relative: *{prediction.closest_relative_species}*."
        )
    else:
        st.success(f"**{prediction.species}** (resolved at species level)")

    st.subheader("Resolved taxonomy")
    taxonomy_row = {rank: getattr(prediction, rank) or "—" for rank in _RANKS}
    st.table(pd.DataFrame([taxonomy_row]))

    st.subheader("Per-rank confidence")
    confidence_df = pd.DataFrame(
        {
            "rank": _RANKS,
            "confidence": [prediction.confidence[r] for r in _RANKS],
            "distance": [prediction.distance[r] for r in _RANKS],
        }
    ).set_index("rank")
    st.bar_chart(confidence_df["confidence"])
    st.dataframe(confidence_df, width="stretch")


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
                row[f"{rank}_coverage"] = scores[rank]["coverage"]
                row[f"{rank}_accuracy"] = scores[rank]["accuracy_among_answered"]
            rows.append(row)

        st.dataframe(pd.DataFrame(rows).set_index("system"), width="stretch")


def main() -> None:
    st.set_page_config(page_title="eDNA Biodiversity Classifier", layout="wide")
    st.title("eDNA Biodiversity Classifier")
    st.caption(
        "Instead of failing outright when a read doesn't match any known reference "
        "sequence, this system always returns a usable, ranked answer -- falling back "
        "to genus/family/order and flagging genuinely novel taxa rather than guessing wrong."
    )

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
