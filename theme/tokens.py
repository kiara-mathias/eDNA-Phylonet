"""Shared palette + type tokens for CSS, SVG, and matplotlib.

Keep these in lockstep with ``.streamlit/config.toml`` and ``theme/app.css``.
"""

from __future__ import annotations

SAND = "#F7F4EC"
NAVY = "#0A1F2E"
TEAL = "#1B5E6C"
SEAFOAM = "#4FA8A0"
CORAL = "#E8734A"
INK = "#2C2C2C"
GENUS = "#2A7A82"
SECONDARY = "#E6E2D6"

RANK_BAND = {
    "species": SEAFOAM,
    "genus": GENUS,
    "family": TEAL,
    "order": NAVY,
}
RANK_INK = {
    "species": NAVY,
    "genus": SAND,
    "family": SAND,
    "order": SAND,
}

FONT_SANS = 'Inter, "Segoe UI", sans-serif'
FONT_MONO = '"IBM Plex Mono", ui-monospace, monospace'
FONT_HREF = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
    "&family=Inter:wght@400;500;600;700&display=swap"
)

TREE_REVEAL_MS = 200
TREE_IFRAME_HEIGHT = 680
RANK_LIT = {
    "species": "#7FCFC8",
    "genus": "#3D8F96",
    "family": "#2A7484",
    "order": "#163A4E",
}
RANKS = ("species", "genus", "family", "order")
