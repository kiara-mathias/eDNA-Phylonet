"""Offline unit tests for src.baselines.naive_bayes_baseline.NaiveBayesBaseline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.baselines.naive_bayes_baseline import NaiveBayesBaseline

_TRAIN_ROWS = [
    {"species": "A a", "genus": "A", "family": "FA", "order": "OA", "sequence": "ACGTACGTACGTACGTACGT"},
    {"species": "A a", "genus": "A", "family": "FA", "order": "OA", "sequence": "ACGTACGTACGTACGTACGA"},
    {"species": "B b", "genus": "B", "family": "FB", "order": "OB", "sequence": "TTTTGGGGCCCCTTTTGGGG"},
    {"species": "B b", "genus": "B", "family": "FB", "order": "OB", "sequence": "TTTTGGGGCCCCTTTTGGGC"},
]


def test_fit_predict_returns_full_taxonomy_for_every_query():
    train_df = pd.DataFrame(_TRAIN_ROWS)
    test_df = pd.DataFrame(
        [{"sequence": "ACGTACGTACGTACGTACGT"}, {"sequence": "TTTTGGGGCCCCTTTTGGGG"}]
    )

    model = NaiveBayesBaseline(k=4)
    model.fit(train_df)
    predictions = model.predict(test_df)

    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.species is not None
        assert prediction.genus is not None
        assert prediction.family is not None
        assert prediction.order is not None


def test_predict_gives_full_coverage_regardless_of_query():
    train_df = pd.DataFrame(_TRAIN_ROWS)
    # A query unlike anything in training -- still gets a (possibly wrong)
    # answer, since MultinomialNB has no abstention concept.
    test_df = pd.DataFrame([{"sequence": "AAAAAAAAAAAAAAAAAAAA"}])

    model = NaiveBayesBaseline(k=4)
    model.fit(train_df)
    predictions = model.predict(test_df)

    assert predictions[0].species in ("A a", "B b")


def test_taxonomy_lookup_is_consistent_with_predicted_species():
    train_df = pd.DataFrame(_TRAIN_ROWS)
    test_df = pd.DataFrame([{"sequence": "ACGTACGTACGTACGTACGT"}])

    model = NaiveBayesBaseline(k=4)
    model.fit(train_df)
    prediction = model.predict(test_df)[0]

    expected = {"A a": ("A", "FA", "OA"), "B b": ("B", "FB", "OB")}[prediction.species]
    assert (prediction.genus, prediction.family, prediction.order) == expected


def test_predict_before_fit_raises():
    model = NaiveBayesBaseline(k=4)

    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame([{"sequence": "ACGT"}]))
