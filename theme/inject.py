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
    """Load fonts + widget CSS via ``st.html``.

    Streamlit 1.5x+ sanitizes ``st.markdown(..., unsafe_allow_html=True)``
    and strips ``<style>``/``<link>``, which dumps the CSS onto the page as
    text. ``st.html`` keeps a style-only block in the event container so it
    does not take layout space.
    """
    st.html(f'<style>@import url("{FONT_HREF}");\n{load_css()}</style>')
