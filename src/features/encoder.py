"""Shared ``fit`` / ``transform`` encoder factory.

``HierarchicalFallback`` only sees a dense embedding matrix -- swapping
k-mer+PCA for a learned encoder must not change fallback code. Every
encoder here implements:

- ``fit(sequences, labels=None)``
- ``transform(sequences) -> np.ndarray`` of shape ``(n, d)``
- ``fit_transform(sequences, labels=None)``
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol

import numpy as np


class SequenceEncoder(Protocol):
    def fit(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> "SequenceEncoder": ...

    def transform(self, sequences: Iterable[str]) -> np.ndarray: ...

    def fit_transform(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> np.ndarray: ...


def encoder_name(config: dict[str, Any]) -> str:
    return str(config.get("name", "kmer_pca")).strip().lower()


def build_encoder(config: dict[str, Any]) -> SequenceEncoder:
    """Construct the encoder named in ``config['name']`` (default ``kmer_pca``)."""
    name = encoder_name(config)
    if name in ("kmer_pca", "kmer", "pca"):
        from src.features.kmer_pca import KmerPCAEncoder

        return KmerPCAEncoder(
            k=int(config.get("k", 4)),
            n_components=int(config.get("n_components", 50)),
            random_state=int(config.get("random_state", 42)),
        )
    if name in ("conv_triplet", "cnn", "triplet"):
        from src.features.conv_triplet import ConvTripletEncoder, conv_triplet_params

        return ConvTripletEncoder(**conv_triplet_params(config))
    if name in ("dnabert", "huggingface", "hf"):
        from src.features.dnabert import HuggingFaceDNAEncoder, dnabert_params

        return HuggingFaceDNAEncoder(**dnabert_params(config))
    raise ValueError(
        f"Unknown encoder {name!r}. Expected one of: kmer_pca, conv_triplet, dnabert."
    )
