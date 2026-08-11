"""Offline tests for iNaturalist/GBIF species-image lookup (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from app.species_image import (
    PLACEHOLDER_SVG,
    lookup_species_image,
    reference_taxon_name,
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
        nearest={rank: "Gadus" for rank in _RANKS},
        nearest_species=[
            NeighborHit(label="Gadus morhua", rank="species", distance=0.4, support=3)
        ],
        nearest_genera=[],
    )
    base.update(overrides)
    return FallbackPrediction(**base)


def _json_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_reference_taxon_name_prefers_committed_species():
    assert reference_taxon_name(_prediction(species="Gadus morhua", is_novel=False, predicted_rank="species")) == (
        "Gadus morhua"
    )
    assert reference_taxon_name(_prediction()) == "Gadus morhua"
    assert reference_taxon_name(_prediction(closest_relative_species="", genus="Gadus")) == "Gadus"


def test_lookup_uses_inaturalist_exact_match():
    def http_get(url, **kwargs):
        assert "inaturalist" in url
        return _json_response(
            {
                "results": [
                    {
                        "name": "Gadus morhua",
                        "rank": "species",
                        "default_photo": {
                            "medium_url": "https://inat.example/cod.jpg",
                            "attribution": "(c) Someone, some rights reserved (CC BY)",
                            "license_code": "cc-by",
                        },
                    }
                ]
            }
        )

    image = lookup_species_image("Gadus morhua", http_get=http_get)
    assert image is not None
    assert image.url == "https://inat.example/cod.jpg"
    assert image.source == "iNaturalist"
    assert "CC BY" in image.attribution


def test_lookup_falls_back_to_gbif_when_inat_has_no_photo():
    def http_get(url, **kwargs):
        if "inaturalist" in url:
            return _json_response({"results": [{"name": "Rare fish", "default_photo": None}]})
        if "species/match" in url:
            return _json_response({"usageKey": 123, "matchType": "EXACT", "canonicalName": "Rare fish"})
        if "/media" in url:
            return _json_response(
                {
                    "results": [
                        {
                            "identifier": "https://gbif.example/rare.jpg",
                            "rightsHolder": "Museum",
                            "license": "http://creativecommons.org/licenses/by/4.0/",
                        }
                    ]
                }
            )
        raise AssertionError(url)

    image = lookup_species_image("Rare fish", http_get=http_get)
    assert image is not None
    assert image.source == "GBIF"
    assert image.url == "https://gbif.example/rare.jpg"
    assert image.attribution == "Museum"


def test_lookup_uses_gbif_occurrence_when_species_media_empty():
    def http_get(url, **kwargs):
        if "inaturalist" in url:
            return _json_response({"results": []})
        if "species/match" in url:
            return _json_response({"usageKey": 9, "matchType": "FUZZY", "canonicalName": "X x"})
        if "/media" in url:
            return _json_response({"results": []})
        if "occurrence" in url:
            return _json_response(
                {"results": [{"media": [{"identifier": "https://gbif.example/occ.jpg", "creator": "A. Diver"}]}]}
            )
        raise AssertionError(url)

    image = lookup_species_image("X x", http_get=http_get)
    assert image is not None
    assert image.url.endswith("occ.jpg")
    assert image.attribution == "A. Diver"


def test_lookup_returns_none_when_both_sources_empty():
    def http_get(url, **kwargs):
        if "inaturalist" in url:
            return _json_response({"results": []})
        if "species/match" in url:
            return _json_response({"matchType": "NONE"})
        raise AssertionError(url)

    assert lookup_species_image("Unknownus nobody", http_get=http_get) is None


def test_lookup_returns_none_on_http_error():
    def http_get(url, **kwargs):
        return _json_response({}, status=503)

    assert lookup_species_image("Gadus morhua", http_get=http_get) is None


def test_lookup_skips_blank_names():
    assert lookup_species_image("  ", http_get=lambda *a, **k: pytest.fail("should not fetch")) is None


def test_placeholder_is_an_inline_silhouette():
    assert "<svg" in PLACEHOLDER_SVG
    assert "No reference photo" in PLACEHOLDER_SVG
