"""Offline unit tests for bootstrap CIs in src.eval.benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.benchmark import _score


@dataclass
class _FakePrediction:
    species: str | None
    genus: str | None = None
    family: str | None = None
    order: str | None = None


def test_bootstrap_ci_brackets_perfect_accuracy_point_estimate():
    preds = [_FakePrediction(species="A") for _ in range(40)] + [_FakePrediction(species="B") for _ in range(10)]
    test_df = pd.DataFrame({"species": ["A"] * 40 + ["B"] * 10})

    rng = np.random.default_rng(0)
    scores = _score(preds, test_df, "species", n_bootstrap=200, ci_level=0.95, rng=rng)

    assert scores["coverage"] == 1.0
    assert scores["accuracy_among_answered"] == 1.0
    assert scores["coverage_ci"] is not None
    assert scores["accuracy_among_answered_ci"] is not None
    lo, hi = scores["accuracy_among_answered_ci"]
    assert lo <= 1.0 <= hi
    assert lo == hi == 1.0  # every resample is still perfect


def test_bootstrap_ci_has_width_when_accuracy_is_noisy():
    # Half correct at species level → CI should have positive width.
    preds = [_FakePrediction(species="A" if i % 2 == 0 else "WRONG") for i in range(100)]
    test_df = pd.DataFrame({"species": ["A"] * 100})

    rng = np.random.default_rng(1)
    scores = _score(preds, test_df, "species", n_bootstrap=300, ci_level=0.95, rng=rng)

    assert scores["accuracy_among_answered"] == 0.5
    lo, hi = scores["accuracy_among_answered_ci"]
    assert lo is not None and hi is not None
    assert lo < 0.5 < hi


def test_bootstrap_handles_unanswered_predictions():
    preds = [_FakePrediction(species=None) for _ in range(20)]
    test_df = pd.DataFrame({"species": ["A"] * 20})

    rng = np.random.default_rng(2)
    scores = _score(preds, test_df, "species", n_bootstrap=50, ci_level=0.95, rng=rng)

    assert scores["coverage"] == 0.0
    assert scores["accuracy_among_answered"] is None
    assert scores["coverage_ci"] == [0.0, 0.0]
    assert scores["accuracy_among_answered_ci"] == [None, None]


def test_bootstrap_disabled_when_n_bootstrap_zero():
    preds = [_FakePrediction(species="A") for _ in range(10)]
    test_df = pd.DataFrame({"species": ["A"] * 10})

    scores = _score(preds, test_df, "species", n_bootstrap=0, ci_level=0.95, rng=None)

    assert scores["coverage"] == 1.0
    assert scores["coverage_ci"] is None
    assert scores["accuracy_among_answered_ci"] is None
