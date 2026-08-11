"""Family silhouette mapping for the Identify result row."""

from __future__ import annotations

from app.family_icons import (
    FAMILY_PATHS,
    KNOWN_FAMILIES,
    family_for_icon,
    family_icon_svg,
    result_callout_html,
)
from src.fallback.novelty import FallbackPrediction, NeighborHit

_RANKS = ("species", "genus", "family", "order")


def _prediction(**overrides) -> FallbackPrediction:
    base = dict(
        predicted_rank="genus",
        is_novel=True,
        closest_relative_species="Gadus morhua",
        species=None,
        genus="Gadus",
        family="Gadidae",
        order="Gadiformes",
        confidence={rank: 0.5 for rank in _RANKS},
        distance={rank: 0.4 for rank in _RANKS},
        support={rank: 3 for rank in _RANKS},
        nearest={rank: "Gadus" if rank != "family" else "Gadidae" for rank in _RANKS},
        nearest_species=[
            NeighborHit(label="Gadus morhua", rank="species", distance=0.4, support=3)
        ],
        nearest_genera=[],
    )
    base.update(overrides)
    return FallbackPrediction(**base)


def test_each_known_family_has_a_distinct_path():
    paths = [FAMILY_PATHS[name] for name in KNOWN_FAMILIES]
    assert len(set(paths)) == len(KNOWN_FAMILIES)
    assert "default" in FAMILY_PATHS


def test_icon_data_family_matches_known_or_default():
    svg = family_icon_svg("Scombridae")
    assert 'data-family="Scombridae"' in svg
    assert FAMILY_PATHS["Scombridae"] in svg
    assert family_icon_svg("NotAFamily").count('data-family="default"') == 1
    assert family_icon_svg(None).count('data-family="default"') == 1


def test_callout_puts_icon_next_to_novel_copy():
    html = result_callout_html(_prediction())
    assert "family-icon" in html
    assert 'data-family="Gadidae"' in html
    assert "Novel taxon" in html
    assert "Gadus morhua" in html
    assert "is-novel" in html


def test_callout_for_species_resolution_uses_seafoam_path():
    html = result_callout_html(
        _prediction(
            predicted_rank="species",
            is_novel=False,
            species="Gadus morhua",
        )
    )
    assert "Resolved at species" in html
    assert "is-resolved" in html
    assert "Gadus morhua" in html


def test_family_for_icon_falls_back_to_nearest():
    prediction = _prediction(family=None, nearest={"family": "Carangidae"})
    assert family_for_icon(prediction) == "Carangidae"
