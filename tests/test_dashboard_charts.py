"""Offline tests for dashboard calibration / FCW helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from app.dashboard_charts import (
    fcw_at_threshold,
    fcw_curve_frame,
    format_sequence_html,
    live_reliability_reports,
    reliability_frame,
)
from app.gallery_examples import BLAST_LIE_EXAMPLES, gallery_examples


def test_gallery_has_three_to_five_held_out_contrasts():
    examples = gallery_examples()
    assert 3 <= len(examples) <= 5
    assert examples == BLAST_LIE_EXAMPLES
    for example in examples:
        assert example.holdout_fraction == pytest.approx(0.5)
        assert example.blast_wrong_at_species
        assert example.blast_wrong_at_genus
        assert example.ours_is_novel
        assert example.ours_species is None
        assert example.blast_pident >= 97.0
        assert example.nearest_species
        assert example.nearest_genera
        assert example.true_genus not in {hit.genus for hit in example.nearest_genera}


def test_reliability_frame_drops_empty_bins():
    reports = [
        {
            "rank": "species",
            "ece": 0.12,
            "bins": [
                {"count": 0, "mean_confidence": float("nan"), "accuracy": float("nan")},
                {"count": 4, "mean_confidence": 0.8, "accuracy": 0.5},
            ],
        }
    ]
    frame = reliability_frame(reports)
    assert list(frame["mean_confidence"]) == [0.8]
    assert list(frame["accuracy"]) == [0.5]


def test_live_reliability_reports_match_bin_counts():
    df = pd.DataFrame(
        {
            "confidence_species": [0.1, 0.2, 0.8, 0.9],
            "correct_species": [0, 0, 1, 1],
            "confidence_genus": [0.2, 0.3, 0.4, 0.5],
            "correct_genus": [1, 0, 1, 0],
            "confidence_family": [0.9, 0.9, 0.9, 0.9],
            "correct_family": [1, 1, 1, 1],
            "confidence_order": [0.9, 0.9, 0.9, 0.9],
            "correct_order": [1, 1, 1, 1],
        }
    )
    reports = live_reliability_reports(df, n_bins=2)
    assert [r.rank for r in reports] == ["species", "genus", "family", "order"]
    assert reports[0].n == 4


def test_fcw_curve_frame_and_threshold_interpolation():
    payload = {
        "table_holdout_fraction": 0.5,
        "splits": [
            {
                "holdout_fraction": 0.5,
                "thresholds": [0.0, 1.0],
                "systems": {
                    "ours": {"novel": {"species": {"fcw_curve": [0.4, 0.0]}}},
                    "blast": {"novel": {"species": {"fcw_curve": [0.8, 0.2]}}},
                    "naive_bayes": None,
                },
            }
        ],
    }
    frame = fcw_curve_frame(payload, rank="species", subset="novel", methods=("ours", "blast"))
    assert set(frame["method"]) == {"ours", "blast"}
    at_half = fcw_at_threshold(frame, 0.5)
    ours = float(at_half.loc[at_half["method"] == "ours", "fcw"].iloc[0])
    blast = float(at_half.loc[at_half["method"] == "blast", "fcw"].iloc[0])
    assert ours == pytest.approx(0.2)
    assert blast == pytest.approx(0.5)


def test_format_sequence_html_colors_bases_and_truncates():
    html = format_sequence_html("ACGTN" + "A" * 200, max_len=10)
    assert "edna-seq" in html
    assert "#1b7a3d" in html  # A
    assert "+195 bp" in html
