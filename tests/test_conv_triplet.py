"""Offline tests for the numpy 1D CNN triplet encoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.conv_triplet import ConvTripletEncoder, one_hot_encode
from src.features.encoder import build_encoder


def _motif_sequences() -> tuple[list[str], list[str]]:
    # Two species with very different motifs so triplet loss has an easy job.
    species_a = ["ACGT" * 16 + "AAAA" for _ in range(4)]
    species_b = ["TGCA" * 16 + "TTTT" for _ in range(4)]
    seqs = species_a + species_b
    labels = ["A a"] * 4 + ["B b"] * 4
    return seqs, labels


def _tiny_encoder(**overrides) -> ConvTripletEncoder:
    params = dict(
        embedding_dim=8,
        n_filters=4,
        kernel_size=5,
        max_len=48,
        epochs=4,
        batch_size=4,
        learning_rate=0.08,
        margin=0.3,
        max_train_sequences=20,
        random_state=0,
    )
    params.update(overrides)
    return ConvTripletEncoder(**params)


def test_one_hot_encode_shape_and_acgt():
    x = one_hot_encode(["ACGTN"], max_len=6)
    assert x.shape == (1, 4, 6)
    assert x[0, 0, 0] == 1  # A
    assert x[0, 1, 1] == 1  # C
    assert x[0, 2, 2] == 1  # G
    assert x[0, 3, 3] == 1  # T
    assert np.all(x[0, :, 4] == 0)  # N
    assert np.all(x[0, :, 5] == 0)  # pad


def test_fit_transform_shape_and_l2_norm():
    seqs, labels = _motif_sequences()
    encoder = _tiny_encoder()
    emb = encoder.fit_transform(seqs, labels)
    assert emb.shape == (len(seqs), 8)
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)


def test_transform_before_fit_raises():
    encoder = _tiny_encoder()
    with pytest.raises(RuntimeError):
        encoder.transform(["ACGT" * 10])


def test_fit_without_labels_raises():
    encoder = _tiny_encoder()
    with pytest.raises(ValueError, match="labels"):
        encoder.fit(["ACGT" * 10, "TGCA" * 10])


def test_deterministic_with_fixed_seed():
    seqs, labels = _motif_sequences()
    a = _tiny_encoder(random_state=1).fit_transform(seqs, labels)
    b = _tiny_encoder(random_state=1).fit_transform(seqs, labels)
    np.testing.assert_allclose(a, b)


def test_same_species_closer_than_other_species_after_fit():
    seqs, labels = _motif_sequences()
    emb = _tiny_encoder().fit_transform(seqs, labels)
    within = np.linalg.norm(emb[0] - emb[1])
    between = np.linalg.norm(emb[0] - emb[4])
    assert within < between


def test_build_encoder_conv_triplet_from_config():
    encoder = build_encoder(
        {
            "name": "conv_triplet",
            "random_state": 0,
            "conv_triplet": {
                "embedding_dim": 6,
                "n_filters": 4,
                "kernel_size": 3,
                "max_len": 32,
                "epochs": 1,
                "batch_size": 4,
            },
        }
    )
    seqs, labels = _motif_sequences()
    emb = encoder.fit_transform(seqs, labels)
    assert emb.shape == (len(seqs), 6)


def test_build_encoder_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown encoder"):
        build_encoder({"name": "not_a_real_encoder"})


def test_build_encoder_default_is_kmer_pca():
    encoder = build_encoder({"k": 3, "n_components": 2, "random_state": 0})
    emb = encoder.fit_transform(["ACGTACGTACGT", "TGCATGCATGCA", "GGCCGGCCGGCC"])
    assert emb.shape[0] == 3
    assert emb.shape[1] == 2
