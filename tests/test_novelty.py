"""Offline unit tests for src.fallback.novelty.HierarchicalFallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fallback.novelty import HierarchicalFallback, NeighborHit


def _rows(species, genus, family, order, points):
    return [
        {"species": species, "genus": genus, "family": family, "order": order, "lat": 0.0, "lon": 0.0}
        for _ in points
    ]


def _train_and_val():
    # Two well-separated species clusters, plus a bit of jitter for val so
    # percentile calibration has a non-degenerate distribution to work with.
    train_points_a = [(-0.2, -0.2), (0.0, 0.0), (0.2, 0.2), (-0.1, 0.1)]
    train_points_b = [(9.8, 9.8), (10.0, 10.0), (10.2, 10.2), (9.9, 10.1)]
    val_points_a = [(-0.3, 0.1), (0.15, -0.05)]
    val_points_b = [(9.85, 10.15), (10.1, 9.95)]

    train_df = pd.DataFrame(
        _rows("A a", "A", "FA", "OA", train_points_a) + _rows("B b", "B", "FB", "OB", train_points_b)
    )
    train_embeddings = np.array(train_points_a + train_points_b)

    val_df = pd.DataFrame(_rows("A a", "A", "FA", "OA", val_points_a) + _rows("B b", "B", "FB", "OB", val_points_b))
    val_embeddings = np.array(val_points_a + val_points_b)

    return train_df, train_embeddings, val_df, val_embeddings


def _fitted_calibrated(
    rank_percentile: float = 90.0, sample_count_prior: float = 0.0
) -> HierarchicalFallback:
    # Default prior=0 keeps distance-only confidence behavior for the
    # original suite; sample-count deflation is covered by dedicated tests.
    train_df, train_embeddings, val_df, val_embeddings = _train_and_val()
    fb = HierarchicalFallback(
        seq_weight=1.0,
        geo_weight=0.0,
        rank_percentile=rank_percentile,
        sample_count_prior=sample_count_prior,
    )
    fb.fit(train_embeddings, train_df)
    fb.calibrate(val_embeddings, val_df, latlon=None)
    return fb


def test_query_near_known_centroid_gets_confident_species_call():
    fb = _fitted_calibrated()

    predictions = fb.predict(np.array([[0.0, 0.0]]), latlon=None)

    assert predictions[0].predicted_rank == "species"
    assert predictions[0].is_novel is False
    assert predictions[0].species == "A a"
    assert predictions[0].genus == "A"
    assert predictions[0].family == "FA"
    assert predictions[0].order == "OA"
    # (0, 0) is close to but not exactly species A's centroid (mean of its
    # jittered training points), so confidence should be high, not exactly 1.0.
    assert predictions[0].confidence["species"] > 0.5


def test_query_exactly_at_centroid_gets_confidence_one():
    train_df, train_embeddings, val_df, val_embeddings = _train_and_val()
    fb = HierarchicalFallback(seq_weight=1.0, geo_weight=0.0, sample_count_prior=0.0)
    fb.fit(train_embeddings, train_df)
    fb.calibrate(val_embeddings, val_df, latlon=None)

    species_a_mask = (train_df["species"] == "A a").to_numpy()
    centroid_a = train_embeddings[species_a_mask].mean(axis=0)

    predictions = fb.predict(np.array([centroid_a]), latlon=None)

    assert predictions[0].confidence["species"] == pytest.approx(1.0, abs=1e-9)


def test_low_sample_count_deflates_confidence_vs_well_sampled_class():
    # Same geometry, unequal support: singleton vs many-sample centroid.
    # At each class's own centroid (distance confidence = 1), reported
    # confidence should equal n/(n+prior).
    prior = 5.0
    train_points_a = [(0.0, 0.0)] * 10
    train_points_b = [(10.0, 10.0)]
    train_df = pd.DataFrame(
        _rows("A a", "A", "FA", "OA", train_points_a) + _rows("B b", "B", "FB", "OB", train_points_b)
    )
    train_embeddings = np.array(train_points_a + train_points_b)
    val_df = train_df.copy()
    val_embeddings = train_embeddings.copy()

    fb = HierarchicalFallback(seq_weight=1.0, geo_weight=0.0, sample_count_prior=prior)
    fb.fit(train_embeddings, train_df)
    fb.calibrate(val_embeddings, val_df, latlon=None)

    pred_a = fb.predict(np.array([[0.0, 0.0]]), latlon=None)[0]
    pred_b = fb.predict(np.array([[10.0, 10.0]]), latlon=None)[0]

    assert pred_a.support["species"] == 10
    assert pred_b.support["species"] == 1
    assert pred_a.confidence["species"] == pytest.approx(10.0 / (10.0 + prior), abs=1e-9)
    assert pred_b.confidence["species"] == pytest.approx(1.0 / (1.0 + prior), abs=1e-9)
    assert pred_b.confidence["species"] < pred_a.confidence["species"]


def test_query_far_from_everything_is_flagged_novel_but_reports_closest_relative():
    fb = _fitted_calibrated()

    predictions = fb.predict(np.array([[1e6, 1e6]]), latlon=None)

    assert predictions[0].is_novel is True
    assert predictions[0].predicted_rank in (None, "genus", "family", "order")
    # Even when fully novel, a "closest relative" is always reported.
    assert predictions[0].closest_relative_species in ("A a", "B b")
    assert predictions[0].species is None
    assert predictions[0].nearest_species[0].label == predictions[0].closest_relative_species
    species_labels = {hit.label for hit in predictions[0].nearest_species}
    assert species_labels == {"A a", "B b"}
    genera_labels = {hit.label for hit in predictions[0].nearest_genera}
    assert genera_labels == {"A", "B"}


def test_calibrated_thresholds_are_positive_for_every_rank():
    # Thresholds are NOT forced monotonic across ranks -- each rank has its
    # own independent scale (median between-centroid distance at that
    # rank), so raw threshold magnitudes aren't on a shared unit. Just
    # confirm calibration produces a sane (positive, finite) value per rank.
    fb = _fitted_calibrated()

    for rank in ("species", "genus", "family", "order"):
        assert fb.thresholds_[rank] > 0
        assert np.isfinite(fb.thresholds_[rank])


def test_confidence_values_are_always_clipped_to_unit_interval():
    fb = _fitted_calibrated()

    queries = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [1e6, -1e6]])
    predictions = fb.predict(queries, latlon=None)

    for prediction in predictions:
        for value in prediction.confidence.values():
            assert 0.0 <= value <= 1.0


def test_calibrate_before_fit_raises():
    fb = HierarchicalFallback()
    _, _, val_df, val_embeddings = _train_and_val()

    with pytest.raises(RuntimeError):
        fb.calibrate(val_embeddings, val_df, latlon=None)


def test_predict_before_calibrate_raises():
    fb = HierarchicalFallback()
    train_df, train_embeddings, _, _ = _train_and_val()
    fb.fit(train_embeddings, train_df)

    with pytest.raises(RuntimeError):
        fb.predict(np.array([[0.0, 0.0]]))


def test_n_nearest_relatives_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_nearest_relatives"):
        HierarchicalFallback(n_nearest_relatives=0)


def test_novel_query_returns_top_k_species_and_genera_in_distance_order():
    # Five species on a line, two genera. Query sits left of the first
    # centroid so nearest-species order is A, B, C, D, E.
    train_points = {
        "A a": [(-0.1, 0.0), (0.0, 0.0), (0.1, 0.0)],
        "B b": [(0.9, 0.0), (1.0, 0.0), (1.1, 0.0)],
        "C c": [(1.9, 0.0), (2.0, 0.0), (2.1, 0.0)],
        "D d": [(3.9, 0.0), (4.0, 0.0), (4.1, 0.0)],
        "E e": [(7.9, 0.0), (8.0, 0.0), (8.1, 0.0)],
    }
    taxonomy = {
        "A a": ("Near", "FA", "OA"),
        "B b": ("Near", "FA", "OA"),
        "C c": ("Near", "FA", "OA"),
        "D d": ("Far", "FB", "OB"),
        "E e": ("Far", "FB", "OB"),
    }
    train_rows = []
    train_embeddings = []
    for species, points in train_points.items():
        genus, family, order = taxonomy[species]
        train_rows.extend(_rows(species, genus, family, order, points))
        train_embeddings.extend(points)
    train_df = pd.DataFrame(train_rows)
    train_embeddings = np.array(train_embeddings)
    val_df = train_df.copy()
    val_embeddings = train_embeddings.copy()

    fb = HierarchicalFallback(
        seq_weight=1.0,
        geo_weight=0.0,
        rank_percentile=90.0,
        sample_count_prior=0.0,
        n_nearest_relatives=5,
    )
    fb.fit(train_embeddings, train_df)
    fb.calibrate(val_embeddings, val_df, latlon=None)

    pred = fb.predict(np.array([[-100.0, 0.0]]), latlon=None)[0]

    assert pred.is_novel is True
    assert [hit.label for hit in pred.nearest_species] == ["A a", "B b", "C c", "D d", "E e"]
    species_dists = [hit.distance for hit in pred.nearest_species]
    assert species_dists == sorted(species_dists)
    assert all(isinstance(hit, NeighborHit) for hit in pred.nearest_species)
    assert pred.nearest_species[0].genus == "Near"
    assert pred.nearest_species[-1].genus == "Far"
    assert pred.closest_relative_species == "A a"
    assert pred.nearest["species"] == "A a"

    assert [hit.label for hit in pred.nearest_genera] == ["Near", "Far"]
    genera_dists = [hit.distance for hit in pred.nearest_genera]
    assert genera_dists == sorted(genera_dists)
    assert pred.nearest_genera[0].family == "FA"


def test_n_nearest_relatives_truncates_and_caps_at_available_labels():
    fb = _fitted_calibrated()
    fb.n_nearest_relatives = 1
    pred = fb.predict(np.array([[0.0, 0.0]]), latlon=None)[0]
    assert len(pred.nearest_species) == 1
    assert len(pred.nearest_genera) == 1
    assert pred.nearest_species[0].label == "A a"

    fb.n_nearest_relatives = 99
    pred = fb.predict(np.array([[0.0, 0.0]]), latlon=None)[0]
    assert len(pred.nearest_species) == 2
    assert len(pred.nearest_genera) == 2


def test_distance_dict_covers_all_ranks():
    fb = _fitted_calibrated()

    predictions = fb.predict(np.array([[0.0, 0.0]]), latlon=None)

    assert set(predictions[0].distance.keys()) == {"species", "genus", "family", "order"}
    assert set(predictions[0].confidence.keys()) == {"species", "genus", "family", "order"}
    assert set(predictions[0].support.keys()) == {"species", "genus", "family", "order"}
    assert all(n >= 1 for n in predictions[0].support.values())
