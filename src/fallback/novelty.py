"""Hierarchical fallback + novelty flagging (Step 5 of the build plan).

Sits after ``Paper2Classifier`` (``src/model/classifier.py``) rather than
replacing it: `Paper2Classifier` always forces a species-level answer,
which is exactly why species/genus accuracy on held-out genera was
correctly 0% in the Step 4 validation. This module instead cascades
species -> genus -> family -> order, stopping at the first rank whose
nearest-centroid distance is within a *calibrated* threshold for that
rank, and otherwise flags the query as novel -- while always reporting the
single nearest known species as a "closest relative" hint, per the
project's "novel taxon, closest relative: X" spec.

Note on distance scaling: unlike ``Paper2Classifier`` (which only needs
the *argmin* across candidates and so can get away with per-query min-max
normalization), this module compares distances against an absolute,
calibrated threshold -- so per-query min-max normalization would be wrong
here: it always rescales the nearest candidate to exactly 0 regardless of
how far away it truly is, which would make every query look "maximally
confident". Instead, each rank's seq/geo distances are divided by a fixed
scale fit once during ``fit()``: the median *pairwise distance between that
rank's own centroids* (i.e. "how far apart are known classes at this rank,
typically"). This -- not median within-class spread -- is deliberate: many
species have only 1-2 training rows, so their within-class spread is
near-zero/unstable and would make the scale wildly noisy; between-centroid
spacing doesn't depend on per-class sample count and stays stable. Each
rank ends up with its own independent scale/unit ("how many typical
between-class gaps away is this query"), so thresholds are *not* forced
monotonic across ranks -- that would incorrectly assume the raw values
share a common unit, which they don't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.classifier import _haversine_km

_RANKS = ("species", "genus", "family", "order")


@dataclass
class FallbackPrediction:
    """One query's hierarchical, novelty-aware prediction.

    ``predicted_rank`` is the deepest rank the model was confident enough
    to commit to (``None`` if even ``order`` failed its threshold).
    ``is_novel`` is ``True`` whenever that isn't ``"species"``. Taxonomy
    fields at or below the unresolved rank are ``None``; fields above the
    resolved rank are still filled in via the training taxonomy (e.g. a
    genus-level call still fills in family/order).
    """

    predicted_rank: str | None
    is_novel: bool
    closest_relative_species: str
    species: str | None
    genus: str | None
    family: str | None
    order: str | None
    confidence: dict[str, float]
    distance: dict[str, float]


@dataclass
class _RankCentroidSet:
    labels: list[str]
    seq_centroids: np.ndarray  # (n_labels, n_dims)
    geo_centroids: np.ndarray  # (n_labels, 2), NaN rows where unavailable
    seq_scale: float  # fixed normalization divisor: median pairwise seq-centroid distance
    geo_scale: float | None  # fixed normalization divisor (km); None if <2 geo centroids at this rank


def _median_pairwise_distance(points: np.ndarray) -> float | None:
    """Median of all pairwise Euclidean distances between rows of ``points``."""
    n = len(points)
    if n < 2:
        return None
    diffs = points[:, None, :] - points[None, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.median(dist_matrix[iu]))


def _median_pairwise_haversine(lat: np.ndarray, lon: np.ndarray) -> float | None:
    n = len(lat)
    if n < 2:
        return None
    dists: list[float] = []
    for i in range(n - 1):
        dists.extend(_haversine_km(lat[i], lon[i], lat[i + 1 :], lon[i + 1 :]).tolist())
    return float(np.median(dists))


class HierarchicalFallback:
    """Cascading, calibrated-threshold novelty-flagging layer.

    Usage: ``fit(train_embeddings, train_df)`` to build per-rank centroids,
    then ``calibrate(val_embeddings, val_df, val_latlon)`` to set confidence
    thresholds from the in-distribution (seen-species) val split, then
    ``predict(embeddings, latlon)``.
    """

    def __init__(self, seq_weight: float = 0.8, geo_weight: float = 0.2, rank_percentile: float = 90.0) -> None:
        total = seq_weight + geo_weight
        self.seq_weight = seq_weight / total
        self.geo_weight = geo_weight / total
        self.rank_percentile = rank_percentile

        self._rank_centroids: dict[str, _RankCentroidSet] = {}
        self._species_taxonomy: dict[str, tuple[str, str, str]] = {}
        self._genus_taxonomy: dict[str, tuple[str, str]] = {}
        self._family_taxonomy: dict[str, str] = {}
        self.thresholds_: dict[str, float] | None = None

    def fit(self, embeddings: np.ndarray, train_df: pd.DataFrame) -> "HierarchicalFallback":
        if len(train_df) != len(embeddings):
            raise ValueError("embeddings and train_df must have the same number of rows")

        df = train_df.reset_index(drop=True)
        has_geo = "lat" in df.columns and "lon" in df.columns

        for rank in _RANKS:
            labels = sorted(df[rank].unique())
            seq_centroids = np.zeros((len(labels), embeddings.shape[1]), dtype=np.float64)
            geo_centroids = np.full((len(labels), 2), np.nan, dtype=np.float64)

            for i, label in enumerate(labels):
                mask = (df[rank] == label).to_numpy()
                seq_centroids[i] = embeddings[mask].mean(axis=0)

                if has_geo:
                    geo_rows = df.loc[mask, ["lat", "lon"]].dropna()
                    if len(geo_rows) > 0:
                        geo_centroids[i] = geo_rows.mean(axis=0).to_numpy()

            seq_scale = max(_median_pairwise_distance(seq_centroids) or 1.0, 1e-9)

            geo_valid = ~np.isnan(geo_centroids).any(axis=1)
            geo_scale = _median_pairwise_haversine(geo_centroids[geo_valid, 0], geo_centroids[geo_valid, 1])
            if geo_scale is not None:
                geo_scale = max(geo_scale, 1e-9)

            self._rank_centroids[rank] = _RankCentroidSet(labels, seq_centroids, geo_centroids, seq_scale, geo_scale)

        for _, row in df.drop_duplicates(subset=["species"]).iterrows():
            self._species_taxonomy[row["species"]] = (row["genus"], row["family"], row["order"])
        for _, row in df.drop_duplicates(subset=["genus"]).iterrows():
            self._genus_taxonomy[row["genus"]] = (row["family"], row["order"])
        for _, row in df.drop_duplicates(subset=["family"]).iterrows():
            self._family_taxonomy[row["family"]] = row["order"]

        return self

    def _nearest_at_rank(
        self, rank: str, embedding: np.ndarray, latlon_row: np.ndarray | None
    ) -> tuple[str, float]:
        centroids = self._rank_centroids[rank]

        seq_dist = np.linalg.norm(centroids.seq_centroids - embedding, axis=1)
        seq_scaled = seq_dist / centroids.seq_scale
        combined = seq_scaled.copy()

        if latlon_row is not None and np.all(np.isfinite(latlon_row)) and centroids.geo_scale is not None:
            geo_mask = ~np.isnan(centroids.geo_centroids).any(axis=1)
            if geo_mask.any():
                geo_dist = _haversine_km(
                    latlon_row[0],
                    latlon_row[1],
                    centroids.geo_centroids[geo_mask, 0],
                    centroids.geo_centroids[geo_mask, 1],
                )
                geo_scaled = geo_dist / centroids.geo_scale
                combined[geo_mask] = self.seq_weight * seq_scaled[geo_mask] + self.geo_weight * geo_scaled

        best_idx = int(np.argmin(combined))
        return centroids.labels[best_idx], float(combined[best_idx])

    def calibrate(
        self,
        embeddings: np.ndarray,
        val_df: pd.DataFrame,
        latlon: np.ndarray | None = None,
    ) -> "HierarchicalFallback":
        """Set per-rank confidence thresholds from the val split.

        ``val_df`` rows should be from species *seen* in training (held
        back from the train split, not from held-out genera) so their
        nearest-distance distribution reflects genuinely in-distribution
        queries.
        """
        if not self._rank_centroids:
            raise RuntimeError("HierarchicalFallback.fit() must be called before calibrate()")

        n = len(val_df)
        distances_per_rank: dict[str, list[float]] = {rank: [] for rank in _RANKS}

        for q in range(n):
            latlon_row = latlon[q] if latlon is not None else None
            for rank in _RANKS:
                _, dist = self._nearest_at_rank(rank, embeddings[q], latlon_row)
                distances_per_rank[rank].append(dist)

        # Each rank's distances are already scaled by that rank's own
        # between-centroid unit (see `_median_pairwise_distance`), so the
        # raw threshold values are not on a shared scale across ranks --
        # deliberately not forced monotonic (see module docstring).
        thresholds = {
            rank: float(np.percentile(distances_per_rank[rank], self.rank_percentile)) if n > 0 else 0.0
            for rank in _RANKS
        }

        self.thresholds_ = thresholds
        return self

    def predict(
        self,
        embeddings: np.ndarray,
        latlon: np.ndarray | None = None,
    ) -> list[FallbackPrediction]:
        if self.thresholds_ is None:
            raise RuntimeError("HierarchicalFallback.calibrate() must be called before predict()")

        predictions: list[FallbackPrediction] = []

        for q in range(len(embeddings)):
            latlon_row = latlon[q] if latlon is not None else None

            nearest: dict[str, tuple[str, float]] = {
                rank: self._nearest_at_rank(rank, embeddings[q], latlon_row) for rank in _RANKS
            }

            predicted_rank: str | None = None
            for rank in _RANKS:
                if nearest[rank][1] <= self.thresholds_[rank]:
                    predicted_rank = rank
                    break

            confidence: dict[str, float] = {}
            for rank in _RANKS:
                threshold = self.thresholds_[rank]
                dist = nearest[rank][1]
                if threshold <= 1e-12:
                    confidence[rank] = 1.0 if dist <= 1e-12 else 0.0
                else:
                    confidence[rank] = float(np.clip(1.0 - dist / threshold, 0.0, 1.0))

            species = genus = family = order = None
            if predicted_rank == "species":
                species = nearest["species"][0]
                genus, family, order = self._species_taxonomy[species]
            elif predicted_rank == "genus":
                genus = nearest["genus"][0]
                family, order = self._genus_taxonomy[genus]
            elif predicted_rank == "family":
                family = nearest["family"][0]
                order = self._family_taxonomy[family]
            elif predicted_rank == "order":
                order = nearest["order"][0]

            predictions.append(
                FallbackPrediction(
                    predicted_rank=predicted_rank,
                    is_novel=(predicted_rank != "species"),
                    closest_relative_species=nearest["species"][0],
                    species=species,
                    genus=genus,
                    family=family,
                    order=order,
                    confidence=confidence,
                    distance={rank: nearest[rank][1] for rank in _RANKS},
                )
            )

        return predictions
