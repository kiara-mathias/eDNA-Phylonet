"""Flat family silhouettes: one accent color, UI-scale, not photos.

Covers the five BOLD families in this dataset. Unknown families fall back
to a generic teleost so the result row always has an icon.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from theme.tokens import CORAL, SEAFOAM

if TYPE_CHECKING:
    from src.fallback.novelty import FallbackPrediction

# viewBox 0 0 120 48 — side-on body shapes, fill-only.
FAMILY_PATHS: dict[str, str] = {
    # Cod: elongate, barbel, truncate tail.
    "Gadidae": (
        "M8 26c2-10 16-16 34-14 16 2 30 4 42 10l24-8v20l-24-6c-12 6-26 8-42 10"
        "C24 40 10 36 8 26zm4 4c-2 4 2 8 6 6"
    ),
    # Tuna/mackerel: torpedo, lunate tail, finlets.
    "Scombridae": (
        "M4 24c10-12 28-16 52-10 14 4 24 6 34 8l22-16-10 16 10 16-22-16"
        "c-10 2-20 4-34 8C32 40 14 36 4 24z"
    ),
    # Flatfish: flattened oval, both eyes on the up-side.
    "Pleuronectidae": (
        "M6 24C10 10 40 4 78 12c16 4 28 8 32 12-4 4-16 8-32 12C40 44 10 38 6 24z"
    ),
    # Grouper/bass: deep body, big head, rounded tail.
    "Serranidae": (
        "M6 24c2-14 18-18 38-14 16 3 28 8 38 14l26-8v16l-26-8c-10 6-22 11-38 14"
        "C24 42 8 38 6 24z"
    ),
    # Jack/trevally: compressed, deeply forked tail, steep head.
    "Carangidae": (
        "M10 24c8-14 26-16 44-8 12 5 22 8 30 8l28-18-12 18 12 18-28-18"
        "c-8 0-18 3-30 8C36 40 18 38 10 24z"
    ),
    "default": (
        "M12 24c6-12 28-16 52-8 10 4 18 6 26 4l22-12v32l-22-12c-8-2-16 0-26 4"
        "C40 40 18 36 12 24z"
    ),
}

KNOWN_FAMILIES = ("Gadidae", "Scombridae", "Pleuronectidae", "Serranidae", "Carangidae")


def family_for_icon(prediction: FallbackPrediction) -> str | None:
    if prediction.family:
        return str(prediction.family)
    nearest = prediction.nearest.get("family")
    return str(nearest) if nearest else None


def family_icon_svg(
    family: str | None,
    *,
    width: int = 72,
    color: str = SEAFOAM,
) -> str:
    """Inline SVG; ``data-family`` is the mapped key (or ``default``)."""
    key = (family or "").strip()
    mapped = key if key in FAMILY_PATHS and key != "default" else "default"
    path = FAMILY_PATHS[mapped]
    height = max(1, round(width * 48 / 120))
    label = html.escape(key or "unknown family")
    return (
        f'<svg class="family-icon" data-family="{html.escape(mapped)}" '
        f'width="{width}" height="{height}" viewBox="0 0 120 48" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{label}">'
        f'<path fill="{color}" d="{path}"/></svg>'
    )


def result_callout_html(prediction: FallbackPrediction) -> str:
    """Identify-tab result row: silhouette + resolved/novel copy."""
    family = family_for_icon(prediction)
    if prediction.is_novel:
        kicker = "Novel taxon"
        detail = str(prediction.closest_relative_species)
        kind = "is-novel"
        kicker_color = CORAL
    else:
        kicker = "Resolved at species"
        detail = str(prediction.species or "—")
        kind = "is-resolved"
        kicker_color = SEAFOAM
    icon = family_icon_svg(family, width=96)
    return (
        f'<div class="result-callout {kind}">'
        f"{icon}"
        f'<div class="result-copy">'
        f'<div class="result-kicker" style="color:{kicker_color}">{html.escape(kicker)}</div>'
        f'<div class="result-detail">{html.escape(detail)}</div>'
        f"</div></div>"
    )
