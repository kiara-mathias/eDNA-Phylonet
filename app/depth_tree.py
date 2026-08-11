"""Taxonomic depth tree: SVG rendered in an iframe, not Streamlit columns.

Four rank nodes sit on a vertical spine. Color is the ocean-depth gradient;
size and opacity follow confidence. Ranks below the trustworthy cutoff fade
and connect with a dashed line into the unresolved zone.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import streamlit.components.v1 as components

from theme.tokens import (
    FONT_HREF,
    FONT_MONO,
    FONT_SANS,
    NAVY,
    RANK_BAND,
    RANK_INK,
    RANK_LIT,
    RANKS,
    SAND,
    SEAFOAM,
    TREE_IFRAME_HEIGHT,
    TREE_REVEAL_MS,
)

if TYPE_CHECKING:
    from src.fallback.novelty import FallbackPrediction

_RANK_INDEX = {rank: i for i, rank in enumerate(RANKS)}
_VIEW_W = 640
_VIEW_H = 660
_CX = 320
_NODE_Y = (44, 158, 272, 386)
_NODE_H = 80
_UNRESOLVED_Y = 510
_UNRESOLVED_H = 112
_MAX_TAXON_CHARS = 28


def rank_visual_state(prediction: FallbackPrediction, rank: str) -> str:
    """``skipped`` (shallower than the commit), ``committed``, or ``abyss`` (deeper)."""
    predicted = prediction.predicted_rank
    if predicted is None:
        return "abyss"
    if _RANK_INDEX[rank] < _RANK_INDEX[predicted]:
        return "skipped"
    if rank == predicted:
        return "committed"
    return "abyss"


def _taxon_at_rank(prediction: FallbackPrediction, rank: str) -> str:
    committed = getattr(prediction, rank)
    if committed:
        return str(committed)
    nearest = prediction.nearest.get(rank)
    return str(nearest) if nearest else "—"


def _clip_taxon(name: str) -> str:
    if len(name) <= _MAX_TAXON_CHARS:
        return name
    return name[: _MAX_TAXON_CHARS - 1].rstrip() + "…"


def _node_width(confidence: float, state: str) -> float:
    conf = max(0.0, min(1.0, confidence))
    width = 220 + 140 * conf
    if state == "committed":
        width = max(width, 300)
    if state == "abyss":
        width = min(width, 280)
    return width


def _end_opacity(confidence: float, state: str) -> float:
    conf = max(0.0, min(1.0, confidence))
    if state == "committed":
        return 1.0
    if state == "skipped":
        return 0.38 + 0.22 * conf
    return 0.18 + 0.14 * conf


def _abyss_note(prediction: FallbackPrediction) -> str:
    stop = prediction.predicted_rank
    if stop is None:
        return "unresolved water — no trustworthy rank"
    return f"below {stop} · not trustworthy"


def build_depth_tree_html(prediction: FallbackPrediction) -> str:
    """Full HTML document for ``st.components.v1.html`` (iframe-local CSS)."""
    nodes_svg: list[str] = []
    connectors_svg: list[str] = []
    states = [rank_visual_state(prediction, rank) for rank in RANKS]
    widths = []
    for rank in RANKS:
        conf = float(prediction.confidence.get(rank, 0.0))
        widths.append(_node_width(conf, rank_visual_state(prediction, rank)))

    for i, rank in enumerate(RANKS[:-1]):
        y1 = _NODE_Y[i] + _NODE_H
        y2 = _NODE_Y[i + 1]
        next_state = states[i + 1]
        dashed = next_state == "abyss"
        dash = 'stroke-dasharray="6 7"' if dashed else ""
        cls = "connector is-abyss" if dashed else "connector"
        connectors_svg.append(
            f'<line class="{cls}" x1="{_CX}" y1="{y1}" x2="{_CX}" y2="{y2}" {dash}/>'
        )

    last_state = states[-1]
    last_bottom = _NODE_Y[-1] + _NODE_H
    dash_into_unresolved = last_state == "abyss" or prediction.predicted_rank != "order"
    dash_attr = 'stroke-dasharray="6 7"' if dash_into_unresolved else ""
    unresolved_cls = "connector is-abyss" if dash_into_unresolved else "connector"
    connectors_svg.append(
        f'<line class="{unresolved_cls}" x1="{_CX}" y1="{last_bottom}" '
        f'x2="{_CX}" y2="{_UNRESOLVED_Y}" {dash_attr}/>'
    )

    for i, rank in enumerate(RANKS):
        state = states[i]
        confidence = float(prediction.confidence.get(rank, 0.0))
        taxon = html.escape(_clip_taxon(_taxon_at_rank(prediction, rank)))
        fg = RANK_INK[rank]
        delay = i * TREE_REVEAL_MS
        width = widths[i]
        x = _CX - width / 2
        y = _NODE_Y[i]
        opacity = _end_opacity(confidence, state)
        filt = 'filter="url(#node-glow)"' if state == "committed" else 'filter="url(#node-shadow)"'
        stroke = (
            f'stroke="{SAND}" stroke-width="1.5"'
            if state == "committed"
            else 'stroke="none"'
        )
        nodes_svg.append(
            f'<g class="node-slot is-{html.escape(state)}" '
            f'style="animation-delay:{delay}ms">'
            f'<g class="node-body" opacity="{opacity:.3f}">'
            f'<rect class="node-plate" x="{x:.1f}" y="{y}" width="{width:.1f}" '
            f'height="{_NODE_H}" rx="10" fill="url(#fill-{rank})" {stroke} {filt}/>'
            f'<text class="rank-name" x="{x + 18:.1f}" y="{y + 22}" fill="{fg}" '
            f'fill-opacity="0.55">{html.escape(rank)}</text>'
            f'<text class="conf" x="{x + width - 16:.1f}" y="{y + 22}" fill="{fg}" '
            f'fill-opacity="0.4" text-anchor="end">{confidence * 100:.0f}%</text>'
            f'<text class="taxon" x="{x + 18:.1f}" y="{y + 52}" fill="{fg}">{taxon}</text>'
            f"</g></g>"
        )

    note = html.escape(_abyss_note(prediction))
    unresolved_delay = 4 * TREE_REVEAL_MS
    rank_fills = "".join(
        f'<linearGradient id="fill-{rank}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{RANK_LIT[rank]}"/>'
        f'<stop offset="100%" stop-color="{RANK_BAND[rank]}"/>'
        f"</linearGradient>"
        for rank in RANKS
    )
    svg = f"""
<svg class="depth-tree" viewBox="0 0 {_VIEW_W} {_VIEW_H}" role="img"
     aria-label="Taxonomic depth tree">
  <defs>
    {rank_fills}
    <linearGradient id="abyss-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{NAVY}" stop-opacity="0.88"/>
      <stop offset="100%" stop-color="{NAVY}" stop-opacity="1"/>
    </linearGradient>
    <filter id="node-shadow" x="-12%" y="-18%" width="124%" height="150%">
      <feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="{NAVY}" flood-opacity="0.28"/>
    </filter>
    <filter id="node-glow" x="-18%" y="-24%" width="136%" height="170%">
      <feDropShadow dx="0" dy="7" stdDeviation="7" flood-color="{SEAFOAM}" flood-opacity="0.42"/>
    </filter>
  </defs>
  <g class="spine">{"".join(connectors_svg)}</g>
  {"".join(nodes_svg)}
  <g class="node-slot unresolved-slot" style="animation-delay:{unresolved_delay}ms">
    <rect class="unresolved" x="80" y="{_UNRESOLVED_Y}" width="480" height="{_UNRESOLVED_H}"
          rx="8" fill="url(#abyss-fill)"/>
    <text class="unresolved-kicker" x="{_CX}" y="{_UNRESOLVED_Y + 44}" text-anchor="middle"
          fill="{SAND}">unresolved</text>
    <text class="unresolved-note" x="{_CX}" y="{_UNRESOLVED_Y + 72}" text-anchor="middle"
          fill="{SAND}">{note}</text>
  </g>
</svg>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="{html.escape(FONT_HREF)}"/>
<style>
  html, body {{
    margin: 0;
    background: {SAND};
    overflow: hidden;
  }}
  .depth-tree {{
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
  }}
  .connector {{
    stroke: {NAVY};
    stroke-width: 2;
    stroke-linecap: round;
    opacity: 0.55;
  }}
  .connector.is-abyss {{
    stroke: {SEAFOAM};
    opacity: 0.28;
  }}
  .node-slot {{
    transform-box: fill-box;
    transform-origin: center;
    animation: tree-reveal 420ms ease-out both;
  }}
  .rank-name, .conf, .unresolved-kicker {{
    font-family: {FONT_MONO};
    font-size: 10px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }}
  .taxon {{
    font-family: {FONT_SANS};
    font-size: 18px;
    font-weight: 600;
  }}
  .unresolved-note {{
    font-family: {FONT_SANS};
    font-size: 15px;
    font-weight: 500;
  }}
  .unresolved-kicker {{
    opacity: 0.55;
    font-size: 10px;
  }}
  .unresolved-note {{
    font-family: {FONT_MONO};
    font-size: 12px;
    letter-spacing: 0.06em;
    opacity: 0.9;
  }}
  @keyframes tree-reveal {{
    from {{ opacity: 0; transform: translateY(-14px); }}
    to {{ opacity: 1; transform: none; }}
  }}
  @media (max-width: 520px) {{
    .taxon {{ font-size: 13px; }}
  }}
</style>
</head>
<body>{svg}</body>
</html>
"""


def render_depth_tree(prediction: FallbackPrediction, *, heading: bool = True) -> None:
    import streamlit as st

    if heading:
        st.markdown(
            '<p class="panel-kicker">Taxonomic depth</p>'
            '<p class="panel-lede">Shallow water is a species-level call. '
            "Each node darker is one rank of fallback. The cascade stops at the "
            "last trustworthy rank; below that is unresolved water.</p>",
            unsafe_allow_html=True,
        )
    components.html(
        build_depth_tree_html(prediction),
        height=TREE_IFRAME_HEIGHT,
        scrolling=False,
    )
