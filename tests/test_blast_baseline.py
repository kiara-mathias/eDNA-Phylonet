"""Offline unit tests for src.baselines.blast_baseline.BlastBaseline.

Functional (subprocess-based) tests are skipped unless ``blastn`` /
``makeblastdb`` are on ``PATH`` -- not installed on this project's Windows
dev machine, but installed in Docker/CI (see ``Dockerfile`` and
``.github/workflows/docker-build.yml``), where they run for real.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.baselines.blast_baseline import BlastBaseline, _write_fasta

pytestmark = pytest.mark.skipif(
    not BlastBaseline.is_available(),
    reason="blastn/makeblastdb not found on PATH (expected on this dev machine; installed in CI)",
)

_TRAIN_ROWS = [
    {
        "process_id": "TRAIN1",
        "species": "A a",
        "genus": "A",
        "family": "FA",
        "order": "OA",
        "sequence": "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    },
    {
        "process_id": "TRAIN2",
        "species": "B b",
        "genus": "B",
        "family": "FB",
        "order": "OB",
        "sequence": "TTTTGGGGCCCCTTTTGGGGCCCCTTTTGGGGCCCCTTTTGGGGCCCCTTTTGGGGCCCC",
    },
]


def test_write_fasta_format(tmp_path):
    df = pd.DataFrame(_TRAIN_ROWS)
    fasta_path = tmp_path / "seqs.fasta"

    _write_fasta(fasta_path, df)

    content = fasta_path.read_text(encoding="utf-8")
    assert ">TRAIN1\n" in content
    assert ">TRAIN2\n" in content
    assert _TRAIN_ROWS[0]["sequence"] in content


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


def test_query_with_no_hit_returns_no_answer(tmp_path):
    train_df = pd.DataFrame(_TRAIN_ROWS)
    # An unrelated, highly divergent sequence relative to both training rows.
    test_df = pd.DataFrame(
        [{"process_id": "QUERY2", "sequence": "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAG"}]
    )

    model = BlastBaseline(db_dir=tmp_path / "blast_db", evalue=1e-10, min_pident=97.0)
    model.fit(train_df)
    predictions = model.predict(test_df)

    assert predictions[0].species is None
    assert predictions[0].genus is None
    assert predictions[0].family is None
    assert predictions[0].order is None


def test_predict_before_fit_raises(tmp_path):
    model = BlastBaseline(db_dir=tmp_path / "blast_db")

    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame([{"process_id": "Q", "sequence": "ACGT"}]))
