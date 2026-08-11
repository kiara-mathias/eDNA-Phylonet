"""Offline unit tests for src.eval.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.calibration import (
    ConfidenceCalibrator,
    apply_rank_calibrators,
    expected_calibration_error,
    fit_rank_calibrators,
    leave_one_holdout_recalibrate,
    reliability_bins,
)


def test_perfectly_calibrated_scores_have_near_zero_ece():
    # Ten equal-sized bins, each with confidence = accuracy = bin midpoint.
    conf = []
    correct = []
    for i in range(10):
        p = (i + 0.5) / 10.0
        conf.extend([p] * 100)
        n_pos = int(round(p * 100))
        correct.extend([1.0] * n_pos + [0.0] * (100 - n_pos))
    ece = expected_calibration_error(np.array(conf), np.array(correct), n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_maximally_miscalibrated_scores_have_high_ece():
    # Always report confidence 0.9 but never correct.
    conf = np.full(200, 0.9)
    correct = np.zeros(200)
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-9)


def test_reliability_bins_count_and_edges():
    conf = np.array([0.05, 0.15, 0.85, 0.95])
    correct = np.array([0.0, 1.0, 1.0, 0.0])
    bins = reliability_bins(conf, correct, n_bins=10)
    assert len(bins) == 10
    assert bins[0].count == 1
    assert bins[1].count == 1
    assert bins[8].count == 1
    assert bins[9].count == 1
    assert bins[0].accuracy == 0.0
    assert bins[1].accuracy == 1.0


def test_isotonic_calibrator_improves_monotone_miscalibration():
    rng = np.random.default_rng(0)
    # Raw scores are systematically overconfident vs. true success rate.
    latent = rng.uniform(0.0, 1.0, size=2000)
    true_p = latent * 0.5
    y = rng.binomial(1, true_p).astype(float)
    raw_ece = expected_calibration_error(latent, y, n_bins=10)

    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(latent, y)
    calibrated = calibrator.predict(latent)
    calibrated_ece = expected_calibration_error(calibrated, y, n_bins=10)
    assert calibrated_ece < raw_ece
    assert calibrated_ece < 0.05


def test_platt_calibrator_maps_to_unit_interval():
    conf = np.linspace(0.0, 1.0, 200)
    y = (conf > 0.4).astype(float)
    calibrator = ConfidenceCalibrator(method="platt")
    calibrator.fit(conf, y)
    out = calibrator.predict(conf)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_fit_and_apply_rank_calibrators():
    confidence_by_rank = {
        "species": np.array([0.1, 0.9, 0.8, 0.2]),
        "genus": np.array([0.2, 0.7, 0.6, 0.3]),
    }
    correct_by_rank = {
        "species": np.array([0.0, 1.0, 1.0, 0.0]),
        "genus": np.array([0.0, 1.0, 1.0, 0.0]),
    }
    calibrators = fit_rank_calibrators(
        confidence_by_rank, correct_by_rank, method="isotonic", ranks=("species", "genus")
    )
    calibrated = apply_rank_calibrators(confidence_by_rank, calibrators)
    assert set(calibrated) == {"species", "genus"}
    assert calibrated["species"].shape == (4,)


def test_leave_one_holdout_recalibrate_reduces_ece():
    # Three holdouts, same underconfident pattern in each.
    rng = np.random.default_rng(1)
    conf_parts = []
    y_parts = []
    group_parts = []
    for g, frac in enumerate([0.3, 0.5, 0.7]):
        conf = rng.uniform(0.0, 0.5, size=400)
        y = (rng.uniform(0.0, 1.0, size=400) < np.clip(conf + 0.4, 0, 1)).astype(float)
        conf_parts.append(conf)
        y_parts.append(y)
        group_parts.append(np.full(400, frac))
    conf = np.concatenate(conf_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    raw_ece = expected_calibration_error(conf, y, n_bins=10)
    calibrated = leave_one_holdout_recalibrate(conf, y, groups, method="isotonic")
    cal_ece = expected_calibration_error(calibrated, y, n_bins=10)
    assert cal_ece < raw_ece
    assert cal_ece < 0.1


def test_degenerate_all_correct_uses_constant():
    conf = np.array([0.2, 0.5, 0.9])
    y = np.ones(3)
    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(conf, y)
    out = calibrator.predict(np.array([0.0, 1.0]))
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0)
