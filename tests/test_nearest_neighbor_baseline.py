"""Offline unit tests for
src.baselines.nearest_neighbor_baseline.NearestNeighborBaseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines.nearest_neighbor_baseline import (
    NearestNeighborBaseline,
    distance_to_confidence,
)

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
    assert predictions[0].nearest["species"] == "A a"
    assert predictions[0].confidence["species"] > 0.9


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
    far = NearestNeighborBaseline()
    far.fit(_TRAIN_EMBEDDINGS, _TRAIN_DF)
    close_conf = far.predict(np.array([[0.0, 0.0]]))[0].confidence["species"]
    far_conf = predictions[0].confidence["species"]
    assert close_conf > far_conf
    assert 0.0 <= far_conf <= 1.0


def test_distance_to_confidence_is_monotone():
    assert distance_to_confidence(0.0, 1.0) == pytest.approx(1.0)
    assert distance_to_confidence(1.0, 1.0) > distance_to_confidence(10.0, 1.0)


def test_fit_rejects_mismatched_lengths():
    model = NearestNeighborBaseline()

    with pytest.raises(ValueError):
        model.fit(np.array([[0.0, 0.0]]), _TRAIN_DF)


def test_predict_before_fit_raises():
    model = NearestNeighborBaseline()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[0.0, 0.0]]))
