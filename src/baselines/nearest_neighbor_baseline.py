"""No-phylogeny nearest-neighbor baseline (Step 6): an ablation of our own
system's encoder.

Uses the exact same k-mer+PCA embeddings ``Paper2Classifier`` does, but
classifies via plain single-nearest-neighbor lookup over *individual*
training rows (no per-species centroid averaging, no taxonomy-hierarchy
reasoning, and no geographic co-occurrence signal). Comparing this against
``Paper2Classifier`` + ``HierarchicalFallback`` isolates exactly what
Paper 2's "phylogeny structure + species co-occurrence" idea contributes,
holding the embedding fixed.

Like ``NaiveBayesBaseline``, this always answers (100% coverage) -- it has
no notion of "I don't know".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import BaselinePrediction

_RANKS = ("species", "genus", "family", "order")


def _median_loo_nn_distance(embeddings: np.ndarray) -> float:
    """Median leave-one-out nearest-neighbor distance on ``embeddings``."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    nn = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist = np.linalg.norm(embeddings - embeddings[i], axis=1)
        dist[i] = np.inf
        nn[i] = float(dist.min())
    med = float(np.median(nn))
    return med if med > 1e-12 else 1.0


def distance_to_confidence(distance: float, scale: float) -> float:
    """Map a nearest-neighbor distance into ``[0, 1]`` via ``exp(-d / scale)``."""
    scale = max(float(scale), 1e-12)
    return float(np.clip(np.exp(-float(distance) / scale), 0.0, 1.0))


class NearestNeighborBaseline:
    def __init__(self) -> None:
        self._train_embeddings: np.ndarray | None = None
        self._train_taxonomy: list[tuple[str, str, str, str]] = []  # aligned with _train_embeddings rows
        self._dist_scale: float = 1.0

    def fit(self, embeddings: np.ndarray, train_df: pd.DataFrame) -> "NearestNeighborBaseline":
        if len(train_df) != len(embeddings):
            raise ValueError("embeddings and train_df must have the same number of rows")

        self._train_embeddings = embeddings
        df = train_df.reset_index(drop=True)
        self._train_taxonomy = list(zip(df["species"], df["genus"], df["family"], df["order"]))
        self._dist_scale = _median_loo_nn_distance(embeddings)
        return self

    def predict(self, embeddings: np.ndarray) -> list[BaselinePrediction]:
        if self._train_embeddings is None:
            raise RuntimeError("NearestNeighborBaseline.fit() must be called before predict()")

        predictions: list[BaselinePrediction] = []
        for q in range(len(embeddings)):
            dist = np.linalg.norm(self._train_embeddings - embeddings[q], axis=1)
            nearest_idx = int(np.argmin(dist))
            species, genus, family, order = self._train_taxonomy[nearest_idx]
            nearest = {"species": species, "genus": genus, "family": family, "order": order}
            conf = distance_to_confidence(float(dist[nearest_idx]), self._dist_scale)
            confidence = {rank: conf for rank in _RANKS}
            predictions.append(
                BaselinePrediction(
                    species=species,
                    genus=genus,
                    family=family,
                    order=order,
                    confidence=confidence,
                    nearest=nearest,
                )
            )

        return predictions
