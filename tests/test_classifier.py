"""Offline unit tests for src.model.classifier.Paper2Classifier.

Uses tiny synthetic embeddings/geography with known geometry so the
nearest-centroid + combined seq/geo distance logic can be checked exactly,
including the missing-geo fallback path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.classifier import Paper2Classifier, _haversine_km, _min_max_normalize


def _train_df():
    # Two well-separated "species" clusters, each with its own genus/family/order.
    return pd.DataFrame(
        [
            {"species": "A a", "genus": "A", "family": "FA", "order": "OA", "lat": 0.0, "lon": 0.0},
            {"species": "A a", "genus": "A", "family": "FA", "order": "OA", "lat": 1.0, "lon": 1.0},
            {"species": "B b", "genus": "B", "family": "FB", "order": "OB", "lat": 50.0, "lon": 50.0},
            {"species": "B b", "genus": "B", "family": "FB", "order": "OB", "lat": 51.0, "lon": 51.0},
        ]
    )


def _train_embeddings():
    # Species A near [0, 0], species B near [10, 10] in embedding space.
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 9.9],
        ]
    )


def test_fit_computes_species_centroids_and_taxonomy():
    clf = Paper2Classifier()
    clf.fit(_train_embeddings(), _train_df())

    assert set(clf.species_names_) == {"A a", "B b"}
    assert clf._taxonomy["A a"] == ("A", "FA", "OA")
    assert clf._taxonomy["B b"] == ("B", "FB", "OB")


def test_predict_picks_nearest_species_by_sequence_distance():
    clf = Paper2Classifier(seq_weight=1.0, geo_weight=0.0)
    clf.fit(_train_embeddings(), _train_df())

    query = np.array([[0.05, 0.05]])  # close to species A's cluster
    predictions = clf.predict(query, latlon=None)

    assert predictions[0].species == "A a"
    assert predictions[0].genus == "A"
    assert predictions[0].family == "FA"
    assert predictions[0].order == "OA"


def test_predict_returns_full_distance_vector_aligned_with_species_names():
    clf = Paper2Classifier()
    clf.fit(_train_embeddings(), _train_df())

    predictions = clf.predict(np.array([[0.0, 0.0]]), latlon=None)

    assert predictions[0].distances.shape == (len(clf.species_names_),)


def test_predict_without_latlon_falls_back_to_sequence_only():
    clf = Paper2Classifier(seq_weight=0.8, geo_weight=0.2)
    clf.fit(_train_embeddings(), _train_df())

    # Query is close to A in embedding space; no lat/lon provided at all.
    predictions = clf.predict(np.array([[0.2, 0.2]]), latlon=None)

    assert predictions[0].species == "A a"


def test_predict_handles_missing_geo_for_query_row():
    clf = Paper2Classifier(seq_weight=0.5, geo_weight=0.5)
    clf.fit(_train_embeddings(), _train_df())

    # latlon array provided but this query's row is NaN -- should fall back
    # to sequence-only distance for that query without raising.
    predictions = clf.predict(np.array([[0.0, 0.0]]), latlon=np.array([[np.nan, np.nan]]))

    assert predictions[0].species == "A a"


def test_geo_signal_can_break_a_near_tie_in_sequence_space():
    clf = Paper2Classifier(seq_weight=0.3, geo_weight=0.7)
    clf.fit(_train_embeddings(), _train_df())

    # Equidistant in embedding space between A's and B's centroids, but
    # geographically much closer to B's centroid (50, 50).
    midpoint = (np.array([0.05, 0.05]) + np.array([10.05, 9.95])) / 2
    predictions = clf.predict(np.array([midpoint]), latlon=np.array([[50.5, 50.5]]))

    assert predictions[0].species == "B b"


def test_predict_before_fit_raises():
    clf = Paper2Classifier()

    with pytest.raises(RuntimeError):
        clf.predict(np.array([[0.0, 0.0]]))


def test_weights_are_normalized_to_sum_one():
    clf = Paper2Classifier(seq_weight=4.0, geo_weight=1.0)

    assert clf.seq_weight == pytest.approx(0.8)
    assert clf.geo_weight == pytest.approx(0.2)


def test_min_max_normalize_handles_constant_input():
    result = _min_max_normalize(np.array([5.0, 5.0, 5.0]))

    assert np.all(result == 0.0)


def test_haversine_zero_distance_for_identical_points():
    dist = _haversine_km(10.0, 20.0, np.array([10.0]), np.array([20.0]))

    assert dist[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance_equator_quarter():
    # Points 90 degrees apart on the equator are a quarter of Earth's
    # circumference apart (~10007.5 km using the classifier's radius).
    dist = _haversine_km(0.0, 0.0, np.array([0.0]), np.array([90.0]))

    assert dist[0] == pytest.approx(10007.5, rel=1e-3)
