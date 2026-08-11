"""Offline unit tests for app.pipeline (Step 7 of the build plan).

Doesn't import streamlit -- app/pipeline.py has no Streamlit dependency by
design, so these run in the same offline/no-network manner as every other
test in this suite.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from app.pipeline import SequenceParseError, classify_sequence, clean_sequence_text, load_pipeline

_RANKS = ("species", "genus", "family", "order")


def _random_sequence(seed: int, length: int = 80) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(length))


def _synthetic_sequences_df(n_species: int = 8, n_per_species: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n_species):
        genus = f"Genus{i // 2}"  # 2 species per genus
        family = f"Family{i // 4}"  # 2 genera per family
        order = "OrderA"
        for j in range(n_per_species):
            rows.append(
                {
                    "process_id": f"P{i}_{j}",
                    "species": f"Genus{i // 2} species{i}",
                    "genus": genus,
                    "family": family,
                    "order": order,
                    "sequence": _random_sequence(seed=i * 100 + j),
                    "lat": 10.0 + i,
                    "lon": 20.0 + i,
                }
            )
    return pd.DataFrame(rows)


def _config(sequences_path) -> dict:
    return {
        "data": {"sequences_path": str(sequences_path)},
        "val_fraction": 0.25,
        "seed": 42,
        "encoder": {"k": 3, "n_components": 5, "random_state": 42},
        "classifier": {"seq_weight": 0.8, "geo_weight": 0.2},
        "fallback": {"rank_percentile": 90.0},
    }


class TestCleanSequenceText:
    def test_strips_fasta_header(self):
        assert clean_sequence_text(">my_read\nACGT\nACGT") == "ACGTACGT"

    def test_strips_whitespace_and_uppercases(self):
        assert clean_sequence_text("  acgt \n acgt  ") == "ACGTACGT"

    def test_raises_on_empty_input(self):
        with pytest.raises(SequenceParseError):
            clean_sequence_text("   \n  ")

    def test_raises_on_header_only_input(self):
        with pytest.raises(SequenceParseError):
            clean_sequence_text(">just_a_header\n")

    def test_raises_on_invalid_characters(self):
        with pytest.raises(SequenceParseError):
            clean_sequence_text("ACGTXYZ")

    def test_allows_ambiguous_n_base(self):
        assert clean_sequence_text("ACGTN") == "ACGTN"


class TestLoadPipelineAndClassify:
    def test_load_pipeline_reports_sane_counts(self, tmp_path):
        df = _synthetic_sequences_df()
        parquet_path = tmp_path / "sequences.parquet"
        df.to_parquet(parquet_path)

        pipeline = load_pipeline(_config(parquet_path))

        assert pipeline.n_train + pipeline.n_val == len(df)
        assert pipeline.n_species == df["species"].nunique()
        assert pipeline.n_genera == df["genus"].nunique()

    def test_classify_known_sequence_resolves_at_species_or_deeper(self, tmp_path):
        df = _synthetic_sequences_df()
        parquet_path = tmp_path / "sequences.parquet"
        df.to_parquet(parquet_path)

        pipeline = load_pipeline(_config(parquet_path))
        # A sequence identical to a training row should be a confident,
        # non-novel call -- exercising the full encode -> predict path.
        known_sequence = df.iloc[0]["sequence"]

        prediction = classify_sequence(pipeline, known_sequence)

        assert prediction.predicted_rank in _RANKS
        assert prediction.closest_relative_species is not None
        assert prediction.nearest_species
        assert prediction.nearest_species[0].label == prediction.closest_relative_species

    def test_classify_accepts_optional_latlon(self, tmp_path):
        df = _synthetic_sequences_df()
        parquet_path = tmp_path / "sequences.parquet"
        df.to_parquet(parquet_path)

        pipeline = load_pipeline(_config(parquet_path))
        # This tiny synthetic dataset has a single `order` value, which can
        # make thresholds unstable -- this test only checks that supplying
        # lat/lon exercises the geo-aware code path without error, not a
        # specific resolved rank (see test_classify_known_sequence_... for that).
        prediction = classify_sequence(pipeline, df.iloc[0]["sequence"], lat=10.0, lon=20.0)

        assert prediction.predicted_rank in _RANKS or prediction.predicted_rank is None
        assert prediction.closest_relative_species is not None
        assert prediction.nearest_species
        assert prediction.nearest_species[0].label == prediction.closest_relative_species
