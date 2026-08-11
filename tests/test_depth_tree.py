"""SVG depth-tree markup follows the cascade commit / abyss states."""

from __future__ import annotations

from app.depth_tree import build_depth_tree_html, rank_visual_state
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
        confidence={"species": 0.12, "genus": 0.71, "family": 0.80, "order": 0.90},
        distance={"species": 1.2, "genus": 0.4, "family": 0.3, "order": 0.2},
        support={"species": 3, "genus": 12, "family": 40, "order": 80},
        nearest={"species": "Gadus morhua", "genus": "Gadus", "family": "Gadidae", "order": "Gadiformes"},
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
        nearest_genera=[],
    )
    base.update(overrides)
    return FallbackPrediction(**base)


def test_abyss_ranks_use_dashed_connectors():
    html = build_depth_tree_html(_prediction())
    assert html.count("is-abyss") >= 2
    assert "stroke-dasharray" in html
    assert "below genus" in html


def test_fully_unresolved_tree_has_no_committed_node():
    html = build_depth_tree_html(
        _prediction(
            predicted_rank=None,
            genus=None,
            family=None,
            order=None,
            confidence={rank: 0.05 for rank in _RANKS},
        )
    )
    assert "is-committed" not in html
    assert "no trustworthy rank" in html
    assert rank_visual_state(
        _prediction(predicted_rank=None, genus=None, family=None, order=None),
        "order",
    ) == "abyss"


def test_species_commit_keeps_lower_ranks_in_abyss():
    prediction = _prediction(
        predicted_rank="species",
        is_novel=False,
        species="Gadus morhua",
        confidence={"species": 0.91, "genus": 0.88, "family": 0.84, "order": 0.80},
    )
    assert rank_visual_state(prediction, "species") == "committed"
    assert rank_visual_state(prediction, "family") == "abyss"
    html = build_depth_tree_html(prediction)
    assert "Gadus morhua" in html
    assert "is-abyss" in html
    assert "linearGradient" in html
    assert "feDropShadow" in html
    assert 'class="taxon"' in html
    assert "fill-opacity=\"0.4\"" in html
