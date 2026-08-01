"""K-mer frequency + PCA sequence encoder (Step 4 of the build plan).

The first, fast-to-validate embedding approach called for in the project
spec: a k-mer frequency vector (no training required to compute) followed
by PCA (fit once on the training split) to a fixed-size dense embedding.
Meant to be swapped for a learned encoder (1D CNN/Transformer) later behind
the same ``fit``/``transform`` interface.
"""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA

_BASES = "ACGT"


class KmerPCAEncoder:
    """Encodes DNA sequences as k-mer frequency vectors, then PCA-reduces them.

    K-mers containing any non-ACGT character (e.g. residual ambiguous
    bases) are skipped when counting, rather than raising -- ``clean.py``
    already caps ambiguous-base fraction, so this only affects a small
    fraction of windows in a small fraction of sequences.
    """

    def __init__(self, k: int = 4, n_components: int = 50, random_state: int = 42) -> None:
        self.k = k
        self.n_components = n_components
        self.random_state = random_state

        self._kmer_index: dict[str, int] = {
            "".join(combo): i for i, combo in enumerate(itertools.product(_BASES, repeat=k))
        }
        self._pca: PCA | None = None

    def _kmer_frequencies(self, sequence: str) -> np.ndarray:
        counts = np.zeros(len(self._kmer_index), dtype=np.float64)
        sequence = sequence.upper()
        n_valid = 0
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i : i + self.k]
            idx = self._kmer_index.get(kmer)
            if idx is not None:
                counts[idx] += 1
                n_valid += 1
        if n_valid > 0:
            counts /= n_valid
        return counts

    def _frequency_matrix(self, sequences: Iterable[str]) -> np.ndarray:
        return np.stack([self._kmer_frequencies(s) for s in sequences])

    def fit(self, sequences: Iterable[str]) -> "KmerPCAEncoder":
        freqs = self._frequency_matrix(sequences)
        # Guard against pathologically small fit sets (e.g. in unit tests)
        # where n_components would otherwise exceed min(n_samples, n_features).
        n_components = min(self.n_components, freqs.shape[0], freqs.shape[1])
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        self._pca.fit(freqs)
        return self

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("KmerPCAEncoder.fit() must be called before transform()")
        freqs = self._frequency_matrix(sequences)
        return self._pca.transform(freqs)

    def fit_transform(self, sequences: Iterable[str]) -> np.ndarray:
        return self.fit(sequences).transform(sequences)
