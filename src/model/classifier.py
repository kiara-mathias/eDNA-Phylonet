"""Base classifier (Step 4 of the build plan): Paper2Classifier.

Implements Paper 2's zero-shot annotation approach -- phylogeny structure
(here, the taxonomic hierarchy: order/family/genus/species, since we don't
have branch-length phylogenies -- see the project's data-source notes) plus
species co-occurrence (here, geographic proximity via lat/lon, the real
signal available in BOLD records) -- as a nearest-centroid classifier.

A query is classified by finding its nearest *known* (training-set)
species centroid under a combined sequence-embedding + geographic distance,
and that species' full taxonomy becomes the prediction at every rank. This
is what makes it zero-shot: a query from a held-out genus still gets a
real, ranked answer, scored at genus/family/order level even though the
exact species call is necessarily wrong. The full per-species distance
vector returned alongside each prediction is intended to feed the (not yet
built) Step 5 fallback/novelty-flagging module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_EARTH_RADIUS_KM = 6371.0
_RANKS = ("species", "genus", "family", "order")


@dataclass
class Prediction:
    """One query's prediction: taxonomy at every rank + the full distance vector."""

    species: str
    genus: str
    family: str
    order: str
    nearest_distance: float
    distances: np.ndarray = field(repr=False)  # aligned with Paper2Classifier.species_names_


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(hi - lo) or hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class Paper2Classifier:
    """Nearest species-centroid classifier combining sequence + geo distance."""

    def __init__(self, seq_weight: float = 0.8, geo_weight: float = 0.2) -> None:
        total = seq_weight + geo_weight
        self.seq_weight = seq_weight / total
        self.geo_weight = geo_weight / total

        self.species_names_: list[str] = []
        self._centroids: np.ndarray | None = None  # (n_species, n_dims)
        self._geo_centroids: np.ndarray | None = None  # (n_species, 2), NaN where unavailable
        self._taxonomy: dict[str, tuple[str, str, str]] = {}  # species -> (genus, family, order)

    def fit(self, embeddings: np.ndarray, train_df: pd.DataFrame) -> "Paper2Classifier":
        """Fit species centroids from training embeddings + metadata.

        ``train_df`` must be row-aligned with ``embeddings`` (same order),
        with columns ``species, genus, family, order`` and optionally
        ``lat, lon``.
        """
        if len(train_df) != len(embeddings):
            raise ValueError("embeddings and train_df must have the same number of rows")

        df = train_df.reset_index(drop=True)
        species_names = sorted(df["species"].unique())

        centroids = np.zeros((len(species_names), embeddings.shape[1]), dtype=np.float64)
        geo_centroids = np.full((len(species_names), 2), np.nan, dtype=np.float64)
        taxonomy: dict[str, tuple[str, str, str]] = {}

        has_geo = "lat" in df.columns and "lon" in df.columns

        for i, species in enumerate(species_names):
            mask = (df["species"] == species).to_numpy()
            centroids[i] = embeddings[mask].mean(axis=0)

            row0 = df.loc[mask].iloc[0]
            taxonomy[species] = (row0["genus"], row0["family"], row0["order"])

            if has_geo:
                geo_rows = df.loc[mask, ["lat", "lon"]].dropna()
                if len(geo_rows) > 0:
                    geo_centroids[i] = geo_rows.mean(axis=0).to_numpy()

        self.species_names_ = species_names
        self._centroids = centroids
        self._geo_centroids = geo_centroids
        self._taxonomy = taxonomy
        return self

    def predict(
        self,
        embeddings: np.ndarray,
        latlon: np.ndarray | None = None,
    ) -> list[Prediction]:
        """Predict taxonomy for each query embedding.

        ``latlon``, if given, is an ``(n, 2)`` array of ``[lat, lon]``,
        with ``NaN`` rows for queries missing coordinates.
        """
        if self._centroids is None:
            raise RuntimeError("Paper2Classifier.fit() must be called before predict()")

        n_queries = len(embeddings)
        predictions: list[Prediction] = []

        for q in range(n_queries):
            seq_dist = np.linalg.norm(self._centroids - embeddings[q], axis=1)
            seq_norm = _min_max_normalize(seq_dist)

            combined = seq_norm.copy()

            if latlon is not None and np.all(np.isfinite(latlon[q])):
                species_geo_mask = ~np.isnan(self._geo_centroids).any(axis=1)
                if species_geo_mask.any():
                    geo_dist = _haversine_km(
                        latlon[q][0],
                        latlon[q][1],
                        self._geo_centroids[species_geo_mask, 0],
                        self._geo_centroids[species_geo_mask, 1],
                    )
                    geo_norm = _min_max_normalize(geo_dist)
                    combined[species_geo_mask] = (
                        self.seq_weight * seq_norm[species_geo_mask] + self.geo_weight * geo_norm
                    )

            best_idx = int(np.argmin(combined))
            species = self.species_names_[best_idx]
            genus, family, order = self._taxonomy[species]

            predictions.append(
                Prediction(
                    species=species,
                    genus=genus,
                    family=family,
                    order=order,
                    nearest_distance=float(combined[best_idx]),
                    distances=combined,
                )
            )

        return predictions
