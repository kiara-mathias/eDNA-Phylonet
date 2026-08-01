"""Offline unit tests for src.features.kmer_pca.KmerPCAEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.kmer_pca import KmerPCAEncoder

SEQUENCES = [
    "ACGTACGTACGTACGTACGTACGT",
    "TTTTACGTACGTACGTACGTACGT",
    "GGGGCCCCACGTACGTACGTACGT",
    "ACGTTTTTGGGGCCCCACGTACGT",
    "CCCCGGGGTTTTACGTACGTACGT",
    "ACGTACGTGGGGCCCCTTTTACGT",
]


def test_fit_transform_shape_matches_n_components():
    encoder = KmerPCAEncoder(k=4, n_components=3, random_state=0)

    embeddings = encoder.fit_transform(SEQUENCES)

    assert embeddings.shape == (len(SEQUENCES), 3)


def test_transform_without_fit_raises():
    encoder = KmerPCAEncoder(k=4, n_components=3, random_state=0)

    with pytest.raises(RuntimeError):
        encoder.transform(SEQUENCES)


def test_deterministic_with_fixed_random_state():
    encoder_a = KmerPCAEncoder(k=4, n_components=3, random_state=0)
    encoder_b = KmerPCAEncoder(k=4, n_components=3, random_state=0)

    emb_a = encoder_a.fit_transform(SEQUENCES)
    emb_b = encoder_b.fit_transform(SEQUENCES)

    np.testing.assert_allclose(emb_a, emb_b)


def test_n_components_capped_by_available_samples_and_features():
    encoder = KmerPCAEncoder(k=2, n_components=1000, random_state=0)

    embeddings = encoder.fit_transform(SEQUENCES[:3])

    # min(n_components=1000, n_samples=3, n_features=4**2=16) == 3
    assert embeddings.shape == (3, 3)


def test_sequence_shorter_than_k_yields_zero_vector():
    encoder = KmerPCAEncoder(k=6, n_components=2, random_state=0)
    short_seq = "ACG"  # shorter than k=6

    freqs = encoder._kmer_frequencies(short_seq)

    assert np.all(freqs == 0)


def test_kmer_frequencies_sum_to_one_for_valid_sequence():
    encoder = KmerPCAEncoder(k=4, n_components=2, random_state=0)

    freqs = encoder._kmer_frequencies("ACGTACGT")

    assert freqs.sum() == pytest.approx(1.0)


def test_kmer_frequencies_skip_ambiguous_bases():
    encoder = KmerPCAEncoder(k=4, n_components=2, random_state=0)

    # "NNNN" windows should be skipped, not counted or error.
    freqs = encoder._kmer_frequencies("ACGTNNNNACGT")

    assert freqs.sum() == pytest.approx(1.0)
