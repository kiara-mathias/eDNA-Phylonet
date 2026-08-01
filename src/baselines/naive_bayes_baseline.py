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

    def fit(self, train_df: pd.DataFrame) -> "NaiveBayesBaseline":
        features = self._freq_encoder._frequency_matrix(train_df["sequence"])
        labels = train_df["species"].to_numpy()

        self._model = MultinomialNB()
        self._model.fit(features, labels)

        for _, row in train_df.drop_duplicates(subset=["species"]).iterrows():
            self._species_taxonomy[row["species"]] = (row["genus"], row["family"], row["order"])

        return self

    def predict(self, test_df: pd.DataFrame) -> list[BaselinePrediction]:
        if self._model is None:
            raise RuntimeError("NaiveBayesBaseline.fit() must be called before predict()")

        features = self._freq_encoder._frequency_matrix(test_df["sequence"])
        predicted_species = self._model.predict(features)

        predictions: list[BaselinePrediction] = []
        for species in predicted_species:
            genus, family, order = self._species_taxonomy[species]
            predictions.append(BaselinePrediction(species=species, genus=genus, family=family, order=order))

        return predictions
