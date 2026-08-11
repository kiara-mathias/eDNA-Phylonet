"""Naive Bayes baseline (Step 6): a stand-in for QIIME2's classifier.

QIIME2's ``classify-sklearn`` naive-Bayes taxonomy classifier works on raw
k-mer/feature counts. This reuses that same idea -- ``MultinomialNB`` on
raw k-mer frequency vectors -- without installing the full QIIME2 conda
stack, which was judged too heavy for this project's portability goals
(see the Step 6 plan).

Like a real ML classifier, this always predicts a species (no abstention
concept) -- 100% coverage by construction, which is exactly the point of
contrast with BLAST (fails outright on divergent sequences) and our own
system's calibrated fallback (abstains/falls back when genuinely unsure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from src.baselines import BaselinePrediction
from src.features.kmer_pca import KmerPCAEncoder


class NaiveBayesBaseline:
    def __init__(self, k: int = 4) -> None:
        self.k = k
        self._freq_encoder = KmerPCAEncoder(k=k)  # only its raw frequency step is used, not PCA
        self._model: MultinomialNB | None = None
        self._species_taxonomy: dict[str, tuple[str, str, str]] = {}
        self._rank_masks: dict[str, dict[str, np.ndarray]] = {}

    def fit(self, train_df: pd.DataFrame) -> "NaiveBayesBaseline":
        features = self._freq_encoder._frequency_matrix(train_df["sequence"])
        labels = train_df["species"].to_numpy()

        self._model = MultinomialNB()
        self._model.fit(features, labels)

        for _, row in train_df.drop_duplicates(subset=["species"]).iterrows():
            self._species_taxonomy[row["species"]] = (row["genus"], row["family"], row["order"])

        classes = list(self._model.classes_)
        genus_of = np.array([self._species_taxonomy[sp][0] for sp in classes])
        family_of = np.array([self._species_taxonomy[sp][1] for sp in classes])
        order_of = np.array([self._species_taxonomy[sp][2] for sp in classes])
        self._rank_masks = {
            "genus": {label: genus_of == label for label in np.unique(genus_of)},
            "family": {label: family_of == label for label in np.unique(family_of)},
            "order": {label: order_of == label for label in np.unique(order_of)},
        }

        return self

    def predict(self, test_df: pd.DataFrame) -> list[BaselinePrediction]:
        if self._model is None:
            raise RuntimeError("NaiveBayesBaseline.fit() must be called before predict()")

        features = self._freq_encoder._frequency_matrix(test_df["sequence"])
        proba = self._model.predict_proba(features)
        classes = self._model.classes_
        predicted_idx = np.argmax(proba, axis=1)

        predictions: list[BaselinePrediction] = []
        for i, idx in enumerate(predicted_idx):
            species = str(classes[idx])
            genus, family, order = self._species_taxonomy[species]
            nearest = {"species": species, "genus": genus, "family": family, "order": order}
            confidence = {
                "species": float(proba[i, idx]),
                "genus": float(proba[i, self._rank_masks["genus"][genus]].sum()),
                "family": float(proba[i, self._rank_masks["family"][family]].sum()),
                "order": float(proba[i, self._rank_masks["order"][order]].sum()),
            }
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
