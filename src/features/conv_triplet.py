"""Small 1D CNN trained with triplet loss (numpy, CPU, no PyTorch).

Same ``fit`` / ``transform`` interface as ``KmerPCAEncoder``. Species labels
are used only during ``fit`` to sample (anchor, positive, negative) triplets;
``HierarchicalFallback`` never sees the network, only the L2-normalized
embeddings.

Kept tiny on purpose: a few filters, global average pool, one linear
projection. Default ``max_train_sequences`` caps how many rows are used for
the triplet loop so a full BOLD table stays CPU-feasible.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def conv_triplet_params(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("conv_triplet") if isinstance(config.get("conv_triplet"), dict) else {}
    merged = {**config, **nested}
    return {
        "embedding_dim": int(merged.get("embedding_dim", 32)),
        "n_filters": int(merged.get("n_filters", 16)),
        "kernel_size": int(merged.get("kernel_size", 7)),
        "max_len": int(merged.get("max_len", 256)),
        "epochs": int(merged.get("epochs", 8)),
        "batch_size": int(merged.get("batch_size", 32)),
        "learning_rate": float(merged.get("learning_rate", 0.05)),
        "margin": float(merged.get("margin", 0.4)),
        "max_train_sequences": int(merged.get("max_train_sequences", 2000)),
        "random_state": int(merged.get("random_state", 42)),
    }


def one_hot_encode(sequences: Sequence[str], max_len: int) -> np.ndarray:
    """``(n, 4, max_len)`` one-hot; non-ACGT and padding stay zero."""
    n = len(sequences)
    out = np.zeros((n, 4, max_len), dtype=np.float64)
    for i, raw in enumerate(sequences):
        seq = raw.upper()[:max_len]
        for j, base in enumerate(seq):
            idx = _BASE_INDEX.get(base)
            if idx is not None:
                out[i, idx, j] = 1.0
    return out


def _conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x ``(B, C_in, L)``, weight ``(C_out, C_in, K)`` → ``(B, C_out, L-K+1)``."""
    window = np.lib.stride_tricks.sliding_window_view(x, weight.shape[2], axis=2)
    # window: (B, C_in, L_out, K); weight: (C_out, C_in, K) → (B, C_out, L_out)
    y = np.einsum("bclk,ock->bol", window, weight)
    return y + bias[None, :, None]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class ConvTripletEncoder:
    """1D conv → ReLU → mean-pool → linear → L2-normalize, triplet-trained."""

    def __init__(
        self,
        embedding_dim: int = 32,
        n_filters: int = 16,
        kernel_size: int = 7,
        max_len: int = 256,
        epochs: int = 8,
        batch_size: int = 32,
        learning_rate: float = 0.05,
        margin: float = 0.4,
        max_train_sequences: int = 2000,
        random_state: int = 42,
    ) -> None:
        if embedding_dim < 1 or n_filters < 1 or kernel_size < 1 or max_len < kernel_size:
            raise ValueError("embedding_dim, n_filters, kernel_size must be >= 1 and max_len >= kernel_size")
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.max_train_sequences = max_train_sequences
        self.random_state = random_state

        self._conv_w: np.ndarray | None = None
        self._conv_b: np.ndarray | None = None
        self._dense_w: np.ndarray | None = None
        self._dense_b: np.ndarray | None = None

    def fit(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> "ConvTripletEncoder":
        if labels is None:
            raise ValueError("ConvTripletEncoder.fit() requires species labels for triplet sampling")
        seqs = list(sequences)
        labs = list(labels)
        if len(seqs) != len(labs):
            raise ValueError("sequences and labels must have the same length")
        if len(seqs) == 0:
            raise ValueError("cannot fit ConvTripletEncoder on an empty collection")

        rng = np.random.default_rng(self.random_state)
        if len(seqs) > self.max_train_sequences:
            idx = rng.choice(len(seqs), size=self.max_train_sequences, replace=False)
            seqs = [seqs[i] for i in idx]
            labs = [labs[i] for i in idx]

        buckets: dict[str, list[int]] = {}
        for i, lab in enumerate(labs):
            buckets.setdefault(lab, []).append(i)
        species_with_pos = [s for s, members in buckets.items() if len(members) >= 2]
        if not species_with_pos or len(buckets) < 2:
            raise ValueError(
                "ConvTripletEncoder needs at least two species and two sequences of one species"
            )

        scale = 1.0 / np.sqrt(4 * self.kernel_size)
        self._conv_w = rng.normal(0.0, scale, size=(self.n_filters, 4, self.kernel_size))
        self._conv_b = np.zeros(self.n_filters, dtype=np.float64)
        pool_dim = self.n_filters
        self._dense_w = rng.normal(0.0, 1.0 / np.sqrt(pool_dim), size=(pool_dim, self.embedding_dim))
        self._dense_b = np.zeros(self.embedding_dim, dtype=np.float64)

        x_all = one_hot_encode(seqs, self.max_len)
        all_species = list(buckets)
        n = len(seqs)
        steps = max(1, int(np.ceil(n / self.batch_size)))

        for _ in range(self.epochs):
            for _step in range(steps):
                anchors, positives, negatives = self._sample_triplets(
                    rng, buckets, species_with_pos, all_species, self.batch_size
                )
                self._sgd_step(x_all[anchors], x_all[positives], x_all[negatives])
        return self

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        if self._conv_w is None:
            raise RuntimeError("ConvTripletEncoder.fit() must be called before transform()")
        x = one_hot_encode(list(sequences), self.max_len)
        emb, _ = self._forward(x)
        return emb

    def fit_transform(self, sequences: Iterable[str], labels: Iterable[str] | None = None) -> np.ndarray:
        seqs = list(sequences)
        return self.fit(seqs, labels).transform(seqs)

    def _sample_triplets(
        self,
        rng: np.random.Generator,
        buckets: dict[str, list[int]],
        species_with_pos: list[str],
        all_species: list[str],
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchors = np.empty(batch_size, dtype=np.int64)
        positives = np.empty(batch_size, dtype=np.int64)
        negatives = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            sp = str(rng.choice(species_with_pos))
            pair = rng.choice(buckets[sp], size=2, replace=False)
            anchors[i] = int(pair[0])
            positives[i] = int(pair[1])
            other = [s for s in all_species if s != sp]
            neg_sp = str(rng.choice(other))
            negatives[i] = int(rng.choice(buckets[neg_sp]))
        return anchors, positives, negatives

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        assert self._conv_w is not None and self._conv_b is not None
        assert self._dense_w is not None and self._dense_b is not None
        conv = _conv1d(x, self._conv_w, self._conv_b)
        hidden = _relu(conv)
        pooled = hidden.mean(axis=2)
        pre = pooled @ self._dense_w + self._dense_b
        emb = _l2_normalize(pre)
        cache = {"x": x, "conv": conv, "hidden": hidden, "pooled": pooled, "pre": pre, "emb": emb}
        return emb, cache

    def _sgd_step(self, x_a: np.ndarray, x_p: np.ndarray, x_n: np.ndarray) -> None:
        assert self._conv_w is not None and self._conv_b is not None
        assert self._dense_w is not None and self._dense_b is not None
        emb_a, cache_a = self._forward(x_a)
        emb_p, cache_p = self._forward(x_p)
        emb_n, cache_n = self._forward(x_n)

        d_ap = np.sum((emb_a - emb_p) ** 2, axis=1)
        d_an = np.sum((emb_a - emb_n) ** 2, axis=1)
        hinge = d_ap - d_an + self.margin
        mask = (hinge > 0).astype(np.float64)[:, None]
        batch = max(len(hinge), 1)

        # dL/demb
        grad_a = (2.0 * (emb_a - emb_p) - 2.0 * (emb_a - emb_n)) * mask / batch
        grad_p = (-2.0 * (emb_a - emb_p)) * mask / batch
        grad_n = (2.0 * (emb_a - emb_n)) * mask / batch

        g_cw = np.zeros_like(self._conv_w)
        g_cb = np.zeros_like(self._conv_b)
        g_dw = np.zeros_like(self._dense_w)
        g_db = np.zeros_like(self._dense_b)
        for grad, cache in ((grad_a, cache_a), (grad_p, cache_p), (grad_n, cache_n)):
            dcw, dcb, ddw, ddb = self._backward(grad, cache)
            g_cw += dcw
            g_cb += dcb
            g_dw += ddw
            g_db += ddb

        lr = self.learning_rate
        self._conv_w -= lr * g_cw
        self._conv_b -= lr * g_cb
        self._dense_w -= lr * g_dw
        self._dense_b -= lr * g_db

    def _backward(
        self, d_emb: np.ndarray, cache: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._dense_w is not None and self._conv_w is not None
        pre = cache["pre"]
        # y = x / ||x||  → Jacobian applied as (I-yy^T)/||x|| * dy
        norms = np.maximum(np.linalg.norm(pre, axis=1, keepdims=True), 1e-9)
        y = cache["emb"]
        d_pre = (d_emb - y * np.sum(d_emb * y, axis=1, keepdims=True)) / norms

        pooled = cache["pooled"]
        d_dense_w = pooled.T @ d_pre
        d_dense_b = d_pre.sum(axis=0)
        d_pooled = d_pre @ self._dense_w.T

        hidden = cache["hidden"]
        length = hidden.shape[2]
        d_hidden = np.repeat(d_pooled[:, :, None] / length, length, axis=2)
        d_conv = d_hidden * (cache["conv"] > 0)

        x = cache["x"]
        window = np.lib.stride_tricks.sliding_window_view(x, self.kernel_size, axis=2)
        d_conv_w = np.einsum("bol,bclk->ock", d_conv, window)
        d_conv_b = d_conv.sum(axis=(0, 2))
        return d_conv_w, d_conv_b, d_dense_w, d_dense_b
