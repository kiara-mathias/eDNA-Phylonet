"""Offline unit tests for
src.baselines.nearest_neighbor_baseline.NearestNeighborBaseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.nearest_neighbor_baseline import NearestNeighborBaseline

_TRAIN_DF = pd.DataFrame(
    [
        {"species": "A a", "genus": "A", "family": "FA", "order": "OA"},
        {"species": "B b", "genus": "B", "family": "FB", "order": "OB"},
    ]
)
_TRAIN_EMBEDDINGS = np.array([[0.0, 0.0], [10.0, 10.0]])


def test_exact_match_query_returns_its_own_taxonomy():
    model = NearestNeighborBaseline()
    model.fit(_TRAIN_EMBEDDINGS, _TRAIN_DF)

    predictions = model.predict(np.array([[0.0, 0.0]]))

    assert predictions[0].species == "A a"
    assert predictions[0].genus == "A"
    assert predictions[0].family == "FA"
    assert predictions[0].order == "OA"


def test_query_closer_to_second_row_picks_it():
    model = NearestNeighborBaseline()
    model.fit(_TRAIN_EMBEDDINGS, _TRAIN_DF)

    predictions = model.predict(np.array([[9.9, 9.9]]))

    assert predictions[0].species == "B b"


def test_always_returns_full_coverage_even_far_away():
    model = NearestNeighborBaseline()
    model.fit(_TRAIN_EMBEDDINGS, _TRAIN_DF)

    predictions = model.predict(np.array([[1e6, -1e6]]))

    assert predictions[0].species is not None


def test_fit_rejects_mismatched_lengths():
    model = NearestNeighborBaseline()

    with pytest.raises(ValueError):
        model.fit(np.array([[0.0, 0.0]]), _TRAIN_DF)


def test_predict_before_fit_raises():
    model = NearestNeighborBaseline()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.0, 0.0]]))
