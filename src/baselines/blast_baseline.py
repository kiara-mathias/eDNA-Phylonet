"""BLAST+ baseline (Step 6): the canonical exact-match-tool comparison.

Wraps real NCBI BLAST+ (``makeblastdb`` / ``blastn``) via ``subprocess`` --
not installed on every machine (in particular, not on this project's
Windows dev machine), so ``is_available()`` lets callers detect and skip
gracefully. It *is* installed in Docker/CI (see ``Dockerfile`` and
``.github/workflows/docker-build.yml``), where this baseline runs for
real.

A query gets **no answer at any rank** if BLAST finds no hit at all, or if
its best hit's percent identity falls below ``min_pident`` (a standard
COI species-level identity cutoff from the barcoding literature) -- this
is the literal "fails outright instead of returning a ranked guess"
behavior this whole project exists to improve on.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.baselines import BaselinePrediction


def _write_fasta(path: Path, df: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            fh.write(f">{row['process_id']}\n{row['sequence']}\n")


class BlastBaseline:
    def __init__(self, db_dir: str | Path, evalue: float = 1e-10, min_pident: float = 97.0) -> None:
        self.db_dir = Path(db_dir)
        self.evalue = evalue
        self.min_pident = min_pident

        self._db_path: Path | None = None
        self._process_id_taxonomy: dict[str, tuple[str, str, str, str]] = {}

    @staticmethod
    def is_available() -> bool:
        return shutil.which("blastn") is not None and shutil.which("makeblastdb") is not None

    def fit(self, train_df: pd.DataFrame) -> "BlastBaseline":
        self.db_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = self.db_dir / "train.fasta"
        _write_fasta(fasta_path, train_df)

        self._db_path = self.db_dir / "train_db"
        subprocess.run(
            ["makeblastdb", "-in", str(fasta_path), "-dbtype", "nucl", "-out", str(self._db_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        for _, row in train_df.iterrows():
            self._process_id_taxonomy[row["process_id"]] = (row["species"], row["genus"], row["family"], row["order"])

        return self

    def predict(self, test_df: pd.DataFrame) -> list[BaselinePrediction]:
        if self._db_path is None:
            raise RuntimeError("BlastBaseline.fit() must be called before predict()")

        query_fasta = self.db_dir / "query.fasta"
        _write_fasta(query_fasta, test_df)

        result = subprocess.run(
            [
                "blastn",
                "-query",
                str(query_fasta),
                "-db",
                str(self._db_path),
                "-outfmt",
                "6 qseqid sseqid pident evalue",
                "-max_target_seqs",
                "1",
                "-evalue",
                str(self.evalue),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # blastn sorts hits per query by descending score, so the first
        # line seen for a given qseqid is its best hit.
        top_hits: dict[str, tuple[str, float]] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            qseqid, sseqid, pident, _evalue = line.split("\t")
            if qseqid not in top_hits:
                top_hits[qseqid] = (sseqid, float(pident))

        predictions: list[BaselinePrediction] = []
        for _, row in test_df.iterrows():
            hit = top_hits.get(row["process_id"])
            if hit is None or hit[1] < self.min_pident:
                predictions.append(BaselinePrediction(species=None, genus=None, family=None, order=None))
                continue

            sseqid, _pident = hit
            species, genus, family, order = self._process_id_taxonomy[sseqid]
            predictions.append(BaselinePrediction(species=species, genus=genus, family=family, order=order))

        return predictions
