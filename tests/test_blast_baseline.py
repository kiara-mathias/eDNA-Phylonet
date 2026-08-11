"""Offline unit tests for src.baselines.blast_baseline.BlastBaseline.

Functional (subprocess-based) tests are skipped unless ``blastn`` /
``makeblastdb`` are on ``PATH`` -- not installed on this project's Windows
dev machine, but installed in Docker/CI (see ``Dockerfile`` and
``.github/workflows/docker-build.yml``), where they run for real.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.baselines.blast_baseline import BlastBaseline, blast_score_to_confidence, _write_fasta

_blast_required = pytest.mark.skipif(
    not BlastBaseline.is_available(),
    reason="blastn/makeblastdb not found on PATH (expected on this dev machine; installed in CI)",
)


def _random_sequence(seed: int, length: int = 150) -> str:
    # Real barcode sequences aren't periodic/low-complexity, so a plain
    # repeated motif (e.g. "ACGTACGT...") gets fully masked by blastn's
    # default DUST low-complexity filter and never produces a hit --
    # pseudo-random, non-repetitive sequences are needed for these tests to
    # exercise real BLAST alignment behavior.
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


_TRAIN_ROWS = [
    {
        "process_id": "TRAIN1",
        "species": "A a",
        "genus": "A",
        "family": "FA",
        "order": "OA",
        "sequence": _random_sequence(seed=1),
    },
    {
        "process_id": "TRAIN2",
        "species": "B b",
        "genus": "B",
        "family": "FB",
        "order": "OB",
        "sequence": _random_sequence(seed=2),
    },
]


def test_blast_score_to_confidence_increases_with_bitscore():
    low = blast_score_to_confidence(bitscore=20.0, evalue=1e-5)
    high = blast_score_to_confidence(bitscore=200.0, evalue=1e-80)
    assert 0.0 <= low < high <= 1.0
    assert blast_score_to_confidence(bitscore=None, evalue=None) == 0.0
    # e-value fallback when bitscore is missing: smaller e-value → higher confidence.
    assert blast_score_to_confidence(None, 1e-20) > blast_score_to_confidence(None, 1.0)


def test_write_fasta_format(tmp_path):
    df = pd.DataFrame(_TRAIN_ROWS)
    fasta_path = tmp_path / "seqs.fasta"

    _write_fasta(fasta_path, df)

    content = fasta_path.read_text(encoding="utf-8")
    assert ">TRAIN1\n" in content
    assert ">TRAIN2\n" in content
    assert _TRAIN_ROWS[0]["sequence"] in content


@_blast_required
def test_fit_and_predict_exact_match(tmp_path):
    train_df = pd.DataFrame(_TRAIN_ROWS)
    test_df = pd.DataFrame(
        [{"process_id": "QUERY1", "sequence": _TRAIN_ROWS[0]["sequence"]}]
    )

    model = BlastBaseline(db_dir=tmp_path / "blast_db", evalue=1.0, min_pident=90.0)
    model.fit(train_df)
    predictions = model.predict(test_df)

    assert predictions[0].species == "A a"
    assert predictions[0].genus == "A"
    assert predictions[0].family == "FA"
    assert predictions[0].order == "OA"
    assert predictions[0].nearest["species"] == "A a"
    assert 0.0 < predictions[0].confidence["species"] <= 1.0


@_blast_required
def test_query_with_no_hit_returns_no_answer(tmp_path):
    train_df = pd.DataFrame(_TRAIN_ROWS)
    # An unrelated, independently-random sequence -- ~25% expected identity
    # by chance, far below any reasonable min_pident/evalue cutoff.
    test_df = pd.DataFrame([{"process_id": "QUERY2", "sequence": _random_sequence(seed=99)}])

    model = BlastBaseline(db_dir=tmp_path / "blast_db", evalue=1e-10, min_pident=97.0)
    model.fit(train_df)
    predictions = model.predict(test_df)

    assert predictions[0].species is None
    assert predictions[0].genus is None
    assert predictions[0].family is None
    assert predictions[0].order is None
    assert predictions[0].confidence["species"] == 0.0
    assert predictions[0].nearest == {}


def test_predict_before_fit_raises(tmp_path):
    model = BlastBaseline(db_dir=tmp_path / "blast_db")

    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame([{"process_id": "Q", "sequence": "ACGT"}]))
