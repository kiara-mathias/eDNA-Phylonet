"""Isolated theme page: background, one input, one button, one tree.

Run with ``streamlit run theme/preview.py`` so CSS and the SVG tree can be
checked without loading the classifier pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.depth_tree import render_depth_tree  # noqa: E402
from app.family_icons import result_callout_html  # noqa: E402
from app.species_image import render_reference_photo  # noqa: E402
from src.fallback.novelty import FallbackPrediction, NeighborHit  # noqa: E402
from theme.inject import inject_theme  # noqa: E402


def _demo_prediction() -> FallbackPrediction:
    return FallbackPrediction(
        predicted_rank="genus",
        is_novel=True,
        closest_relative_species="Gadus morhua",
        species=None,
        genus="Gadus",
        family="Gadidae",
        order="Gadiformes",
        confidence={"species": 0.12, "genus": 0.71, "family": 0.80, "order": 0.90},
        distance={"species": 1.2, "genus": 0.4, "family": 0.3, "order": 0.2},
        support={"species": 3, "genus": 12, "family": 40, "order": 80},
        nearest={
            "species": "Gadus morhua",
            "genus": "Gadus",
            "family": "Gadidae",
            "order": "Gadiformes",
        },
        nearest_species=[
            NeighborHit(
                label="Gadus morhua",
                rank="species",
                distance=1.2,
                support=3,
                genus="Gadus",
                family="Gadidae",
                order="Gadiformes",
            )
        ],
        nearest_genera=[
            NeighborHit(
                label="Gadus",
                rank="genus",
                distance=0.4,
                support=12,
                genus="Gadus",
                family="Gadidae",
                order="Gadiformes",
            )
        ],
    )


def main() -> None:
    st.set_page_config(page_title="eDNA theme preview", layout="wide")
    inject_theme()
    st.markdown(
        """
        <div class="record-header">
          <p class="record-kicker">theme preview</p>
          <p class="record-title">Identify this specimen</p>
          <p class="record-lede">If this looks like default Streamlit, the CSS did not attach.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input("Sequence accession", placeholder="BOLD:AAI1234")
    st.button("Identify this specimen", type="primary")
    st.markdown(result_callout_html(_demo_prediction()), unsafe_allow_html=True)
    st.success("Streamlit alert (error/info still used for validation).")
    st.warning("Unused default warning, for CSS checks.")
    tabs = st.tabs(["Identify", "Evidence", "Method"])
    with tabs[0]:
        tree_col, photo_col = st.columns([1.35, 0.85], vertical_alignment="top")
        with tree_col:
            render_depth_tree(_demo_prediction())
        with photo_col:
            render_reference_photo(_demo_prediction())
    with tabs[1]:
        st.dataframe(pd.DataFrame({"rank": ["species", "genus"], "confidence": [0.12, 0.71]}), hide_index=True)
    with tabs[2]:
        st.info("Widget overrides for tabs, alerts, and tables live in theme/app.css.")


if __name__ == "__main__":
    main()
