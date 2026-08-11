"""Visual tokens and Streamlit theme injection for the specimen record."""

from theme.inject import inject_theme, load_css
from theme.tokens import (
    CORAL,
    INK,
    NAVY,
    RANK_BAND,
    RANK_INK,
    SAND,
    SEAFOAM,
    TEAL,
)

__all__ = [
    "CORAL",
    "INK",
    "NAVY",
    "RANK_BAND",
    "RANK_INK",
    "SAND",
    "SEAFOAM",
    "TEAL",
    "inject_theme",
    "load_css",
]
