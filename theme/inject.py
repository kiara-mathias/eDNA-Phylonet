"""Inject palette CSS into a Streamlit page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from theme.tokens import CORAL, FONT_HREF, INK, NAVY, SAND, SEAFOAM, TEAL

_CSS_PATH = Path(__file__).with_name("app.css")

_ROOT = f"""
:root {{
  --edna-sand: {SAND};
  --edna-navy: {NAVY};
  --edna-teal: {TEAL};
  --edna-seafoam: {SEAFOAM};
  --edna-coral: {CORAL};
  --edna-ink: {INK};
}}
"""


def load_css() -> str:
    return _ROOT + "\n" + _CSS_PATH.read_text(encoding="utf-8")


def inject_theme() -> None:
    """Fonts via <link>, widget overrides via a single style block."""
    st.markdown(
        f'<link rel="preconnect" href="https://fonts.googleapis.com">'
        f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{FONT_HREF}" rel="stylesheet">'
        f"<style>{load_css()}</style>",
        unsafe_allow_html=True,
    )
