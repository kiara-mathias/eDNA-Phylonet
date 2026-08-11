"""Builds and serves the deployed inference pipeline for the Streamlit
dashboard (Step 7 of the build plan).

Deliberately not the same object graph as ``src/eval/benchmark.py``: that
script holds out entire genera to *measure* zero-shot generalization,
while this module fits on (almost) everything known, holding back only a
small random validation slice to calibrate ``HierarchicalFallback``'s
novelty thresholds -- the deployed model should use every genus it has
ever seen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import load_config, resolve_path
from src.fallback.novelty import FallbackPrediction, HierarchicalFallback
from src.features.encoder import SequenceEncoder, build_encoder

_VALID_BASES = set("ACGTN")


@dataclass
class Pipeline:
    encoder: SequenceEncoder
    fallback: HierarchicalFallback
    n_train: int
    n_val: int
    n_species: int
    n_genera: int
    n_families: int
    n_orders: int


class SequenceParseError(ValueError):
    """Raised when pasted/uploaded input has no usable sequence in it."""


def clean_sequence_text(raw_text: str) -> str:
    """Normalize pasted FASTA or raw-sequence text into a bare ACGTN string.

    Strips ``>`` header lines (if any -- a full FASTA record is accepted,
    not just a bare sequence), whitespace, and line breaks, then
    uppercases. Raises ``SequenceParseError`` if nothing usable remains or
    the result contains characters outside ``ACGTN``.
    """
    lines = [line.strip() for line in raw_text.strip().splitlines()]
    sequence_lines = [line for line in lines if line and not line.startswith(">")]
    sequence = "".join(sequence_lines).upper()
    sequence = re.sub(r"\s", "", sequence)

    if not sequence:
        raise SequenceParseError("No sequence found in input -- paste a raw sequence or a FASTA record.")

    invalid_chars = set(sequence) - _VALID_BASES
    if invalid_chars:
        raise SequenceParseError(
            f"Sequence contains characters outside A/C/G/T/N: {''.join(sorted(invalid_chars))}"
        )

    return sequence


def load_pipeline(config: dict[str, Any]) -> Pipeline:
    sequences_df = pd.read_parquet(resolve_path(config["data"]["sequences_path"]))

    train_df, val_df = train_test_split(
        sequences_df,
        test_size=config["val_fraction"],
        random_state=config["seed"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    encoder_cfg = config["encoder"]
    encoder = build_encoder(encoder_cfg)
    train_embeddings = encoder.fit_transform(train_df["sequence"], train_df["species"])
    val_embeddings = encoder.transform(val_df["sequence"])

    classifier_cfg = config["classifier"]
    fallback_cfg = config["fallback"]
    fallback = HierarchicalFallback(
        seq_weight=classifier_cfg["seq_weight"],
        geo_weight=classifier_cfg["geo_weight"],
        rank_percentile=fallback_cfg["rank_percentile"],
        sample_count_prior=float(fallback_cfg.get("sample_count_prior", 5.0)),
    )
    fallback.fit(train_embeddings, train_df)

    val_latlon = _latlon_array(val_df)
    fallback.calibrate(val_embeddings, val_df, val_latlon)

    return Pipeline(
        encoder=encoder,
        fallback=fallback,
        n_train=len(train_df),
        n_val=len(val_df),
        n_species=sequences_df["species"].nunique(),
        n_genera=sequences_df["genus"].nunique(),
        n_families=sequences_df["family"].nunique(),
        n_orders=sequences_df["order"].nunique(),
    )


def _latlon_array(df: pd.DataFrame) -> np.ndarray | None:
    if "lat" not in df.columns or "lon" not in df.columns:
        return None
    return df[["lat", "lon"]].to_numpy(dtype=np.float64)


def classify_sequence(
    pipeline: Pipeline,
    sequence: str,
    lat: float | None = None,
    lon: float | None = None,
) -> FallbackPrediction:
    """Runs one cleaned sequence through the fitted pipeline."""
    embedding = pipeline.encoder.transform([sequence])
    latlon = np.array([[lat, lon]], dtype=np.float64) if lat is not None and lon is not None else None
    return pipeline.fallback.predict(embedding, latlon)[0]


def load_app_config(config_path: str = "configs/app.yaml") -> dict[str, Any]:
    return load_config(config_path)
